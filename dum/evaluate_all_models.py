#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_all_models.py
@Time    :   2024/11/21 11:06:01
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import argparse
import glob
import json
import math
import os
import pickle as pkl
import re
import time

import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import data_utils.ood_detection.caltech256 as caltech256
# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.dtd as dtd
import data_utils.ood_detection.fer2013 as fer2013
import data_utils.ood_detection.gauss as gauss
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.place365 as place365
import data_utils.ood_detection.stl as stl
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet
from metrics.calibration_metrics import expected_calibration_error
# Import metrics to compute
from metrics.classification_metrics import (test_classification_net, test_classification_net_ensemble, test_classification_net_logits)
from metrics.ood_metrics import (get_roc_auc, get_roc_auc_ensemble, get_roc_auc_logits)
from metrics.uncertainty_confidence import (confidence, entropy, logsumexp, maxval, sumexp)
# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.vgg import vgg16
from net.vit import vit
from net.wide_resnet import wrn
from utils.args import eval_args
from utils.ensemble_utils import ensemble_forward_pass, load_ensemble
from utils.eval_utils import model_load_name
# Import GMM utils
from utils.gmm_utils import (get_embeddings, gmm_evaluate, gmm_evaluate_for_adv, gmm_evaluate_with_perturbation,
                             gmm_evaluate_with_perturbation_for_adv, gmm_fit, gradient_norm_collect, maxp_evaluate, maxp_evaluate_with_perturbation)
# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature
from utils.train_utils import model_save_name, seed_torch
from utils.plots_utils import plot_embedding_2d, inter_intra_class_ratio, create_gif_from_images
from utils.normality_test import normality_score

# Dataset params
dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "lsun": 10, "tiny_iamgenet": 200}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "fer2013": fer2013,
    "mnist": mnist,
    "lsun": lsun,
    "dtd": dtd,
    "stl": stl,
    "place365": place365,
    "caltech256": caltech256,
    "tiny_imagenet": tiny_imagenet
}

# Mapping model name to model function
models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}

model_to_num_dim = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048, "wide_resnet": 640, "vgg16": 512, "vit": 768}

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    num_classes = 10
    seed_torch()

    device = "cuda:0"
    model = "resnet50"
    run = "run35"
    method = "univariate"

    dir = f"./saved_models/{run}/{model}_sn_3.0_mod_seed_1/"
    sub_dirs = os.listdir(dir)[:1]
    sub_dirs = ["2024_11_21_23_06_00"]
    for time_str in sub_dirs:
        res = {}
        dir2 = os.path.join(dir, time_str)
        model_save_name = os.path.join(dir2, f"{model}_sn_3.0_mod_seed_1_best.model")
        train_loader, val_loader = dataset_loader["cifar10"].get_train_valid_loader(
            root="./data/",
            batch_size=512,
            augment=False,  #False
            val_seed=1,
            val_size=0.0,
            pin_memory=1,
        )
        train_loader2, _ = dataset_loader["cifar10"].get_train_valid_loader(
            root="./data/",
            batch_size=32,
            augment=False,  #False
            val_seed=1,
            val_size=0.0,
            pin_memory=1,
        )
        test_loader = cifar10.get_test_loader(
            root="./data/",
            batch_size=512,
            sample_size=10000000,
        )
        test_loader2 = cifar10.get_test_loader(
            root="./data/",
            batch_size=32,
            sample_size=1000,
        )

        for epoch in tqdm(range(300, 320)):
            model_save_name = os.path.join(dir2, f"{model}_sn_3.0_mod_seed_1_epoch_{epoch}.model")
            print(f"load {model_save_name}")
            net = models[model](
                spectral_normalization=True,
                mod=True,
                num_classes=num_classes,
                temp=1.0,
            )
            net.to(device)
            net.load_state_dict(torch.load(str(model_save_name), map_location=device), strict=True)
            net.eval()
            (
                conf_matrix,
                accuracy,
                labels_list,
                predictions,
                confidences,
            ) = test_classification_net(net, test_loader, device)

            cache_path = model_save_name.replace('.model', '.cache')
            load_cache = True
            if load_cache and os.path.exists(cache_path):
                print(f"load cache from {cache_path}")
                with open(cache_path, 'rb') as file:
                    cache = pkl.load(file)
                    embeddings = cache["embeddings"].to(device)
                    labels = cache["labels"].to(device)
                    norm_threshold = cache["norm_threshold"]
            else:
                embeddings, labels, norm_threshold = get_embeddings(
                    net,
                    train_loader,
                    num_dim=model_to_num_dim[model],
                    dtype=torch.double,
                    device=device,
                    storage_device=device,
                )
                cache = {"embeddings": embeddings.cpu(), "labels": labels.cpu(), "norm_threshold": norm_threshold}
                with open(cache_path, "wb") as f:
                    pkl.dump(cache, f)

            gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
            logits, labels, preds = gmm_evaluate(
                net,
                gaussians_model,
                test_loader2,
                device=device,
                num_classes=num_classes,
                storage_device=device,
            )
            Xs = []
            ys = []
            for images, labels in test_loader2:
                images = images.to(device)
                _ = net(images)
                embeddings = net.feature
                Xs.append(embeddings.cpu().detach().numpy())
                ys.append(labels.detach().numpy())
            X = np.concatenate(Xs)
            y = np.concatenate(ys)

            auroc = []
            auprc = []
            for ood in ["svhn", "cifar100", "lsun", "mnist", "tiny_imagenet"]:
                ood_test_loader = dataset_loader[ood].get_test_loader(32, root="./data/", sample_size=1000)
                ood_logits, ood_labels, _ = gmm_evaluate(
                    net,
                    gaussians_model,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )
                m1_fpr95, m1_auroc, m1_auprc = get_roc_auc_logits(logits, ood_logits, maxval, device, conf=True)

                Xs_ood = []
                ys_ood = []
                for images, _ in ood_test_loader:
                    labels = np.ones(images.shape[0]) * 10  #标记label=10为OOD样本
                    images = images.to(device)
                    _ = net(images)
                    embeddings = net.feature
                    Xs_ood.append(embeddings.cpu().detach().numpy())
                    ys_ood.append(labels)
                X_ood = np.concatenate(Xs_ood)
                y_ood = np.concatenate(ys_ood)

                X_all = np.concatenate([X, X_ood])
                y_all = np.concatenate([y, y_ood])

                start = time.time()
                tsne = TSNE(n_components=2, init='pca', perplexity=50, random_state=0)
                X_tsne = tsne.fit_transform(X_all)
                print(time.time() - start)

                fig = plot_embedding_2d(X_tsne, y_all, 10, f"epoch:{epoch},auroc:{m1_auroc:.3f},auprc:{m1_auprc:.3f}")
                save_loc = f"./results/{run}/{model}_sn_3.0_mod_seed_1/{time_str}"
                os.makedirs(save_loc, exist_ok=True)
                fig.savefig(os.path.join(save_loc, f"{ood}_stats_{epoch}.jpg"), dpi=100, bbox_inches='tight')

                auroc.append(m1_auroc)
                auprc.append(m1_auprc)
                
                break

            res[epoch] = {}
            res[epoch]["auroc"] = auroc
            res[epoch]["auprc"] = auprc
            print(f"epoch:{epoch},accuracy:{accuracy},auroc:{auroc},auprc:{auprc}")

        save_loc = f"./results/{run}/{model}_sn_3.0_mod_seed_1/{time_str}"
        saved_name = "all_res.json"
        os.makedirs(save_loc, exist_ok=True)
        with open(os.path.join(save_loc, saved_name), "w") as f:
            json.dump(res, f)
            print(f"save to {os.path.join(save_loc,saved_name)}")
