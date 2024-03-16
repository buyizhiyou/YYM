#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   clip_test.py
@Time    :   2024/03/04 14:34:59
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
"""
Script to evaluate a single model. 
"""
import os
import json
import math
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch
import clip
import numpy as np

from torchvision import datasets
from torchvision.transforms import Resize

# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50, resnet101, resnet110, resnet152
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import metrics to compute
from metrics.classification_metrics import (test_classification_net,
                                            test_classification_net_logits,
                                            test_classification_net_ensemble)
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.kde_utils import kde_evaluate, kde_fit
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import eval_args

# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "tiny_iamgenet": 200
}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "tiny_imagenet": tiny_imagenet
}



if __name__ == "__main__":

    args = eval_args().parse_args()

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if cuda else "cpu")

    #load clip model
    model, preprocess = clip.load("RN50", device=device)
    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    data_dir = "./data"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=preprocess,)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=preprocess,)
    train_loader = torch.utils.data_utils.DataLoader(
        train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False,
    )
    test_loader = torch.utils.data_utils.DataLoader(
        test_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False,
    )


    dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=preprocess,)
    ood_test_loader = torch.utils.data_utils.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )

    embeddings = []
    labels = []
    torch_resize = Resize((224,224))
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
        features = model.encode_image(batch_x).to(torch.float32)
        # features /= features.norm(dim=-1, keepdim=True)

        embeddings.append(features.cpu().detach().numpy())
        labels.append(batch_y.cpu().detach().numpy())

    embeddings = np.concatenate(embeddings, 0)
    labels = np.concatenate(labels, 0)

    embeddings = torch.from_numpy(embeddings).to(device)
    labels = torch.from_numpy(labels).to(device)
    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings,
                                          labels=labels,
                                          num_classes=num_classes)

    logits = []
    labels = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
        features = model.encode_image(batch_x).to(torch.float32)
        # features /= features.norm(dim=-1, keepdim=True)
        log_probs = gaussians_model.log_prob(features[:, None, :])

        logits.append(log_probs.cpu().detach().numpy())
        labels.append(batch_y.cpu().detach().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)


    ood_logits = []
    ood_labels = []
    for batch_x, batch_y in ood_test_loader:
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
        features = model.encode_image(batch_x).to(torch.float32)
        # features /= features.norm(dim=-1, keepdim=True)
        log_probs = gaussians_model.log_prob(features[:, None, :])

        ood_logits.append(log_probs.cpu().detach().numpy())
        ood_labels.append(batch_y.cpu().detach().numpy())

    ood_logits = np.concatenate(ood_logits, 0)
    ood_labels = np.concatenate(ood_labels, 0)


    logits = torch.from_numpy(logits).to(device)
    ood_logits = torch.from_numpy(ood_logits).to(device)
    (_, _, _), (_, _,
                _), m1_auroc, m1_auprc = get_roc_auc_logits(logits,
                                                            ood_logits,
                                                            logsumexp,
                                                            device,
                                                            confidence=True)
    (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(
        logits, ood_logits, entropy, device)

    t_m1_auroc = m1_auroc
    t_m1_auprc = m1_auprc
    t_m2_auroc = m2_auroc
    t_m2_auprc = m2_auprc

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["t_m1_auroc"] = t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = t_m2_auprc.item()

    res_dict["info"] = vars(args)

    with open(
            "./logs/res_clip_" + model_save_name(args.model, args.sn, args.mod,
                                                 args.coeff, args.seed,False) + "_" +
            args.model_type + "_" + args.dataset + "_" + args.ood_dataset +
            ".json",
            "w",
    ) as f:
        json.dump(res_dict, f)
