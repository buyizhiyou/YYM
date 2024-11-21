#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   norm_detect_select.py
@Time    :   2024/11/19 17:12:38
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
from net.lenet import lenet
from net.resnet import resnet18, resnet50  # 自己实现的spectral norm
# from net.resnet2 import resnet18, resnet50 #官方实现的spectral norm
from net.vgg import vgg16  # 自己实现的
from net.vit import vit
# from net.vgg2 import vgg16 #官方实现的
from net.wide_resnet import wrn
from utils.normality_test import normality_score
from utils.plots_utils import (create_gif_from_images, inter_intra_class_ratio, plot_embedding_2d)

models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}
train_loader, _ = cifar10.get_train_valid_loader(root="./data/", batch_size=1024, augment=False, val_size=0., val_seed=1, pin_memory=0, contrastive=0)
ood_test_loader = svhn.get_test_loader(512, root="./data/", sample_size=2000)

device = "cuda:0"
model = "vgg16"
run = "run32"
method = "univariate"
net = models[model](
    spectral_normalization=True,
    mod=True,
    num_classes=10,
    temp=1.0,
)
net.to(device)
cudnn.benchmark = True
dir = f"./saved_models/{run}/{model}_sn_3.0_mod_seed_1/"
sub_dirs = os.listdir(dir)
for time_str in sub_dirs:
    dir2 = os.path.join(dir, time_str)
    model_save_name = os.path.join(dir2, f"{model}_sn_3.0_mod_seed_1_best.model")
    net.load_state_dict(torch.load(str(model_save_name), map_location=device), strict=True)
    net.eval()

    Xs = []
    ys = []
    for images, labels in train_loader:
        images = images.to(device)
        _ = net(images)
        embeddings = net.feature
        Xs.append(embeddings.cpu().detach().numpy())
        ys.append(labels.detach().numpy())
    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    _, best_stats = normality_score(X, y, method)
    for epoch in tqdm(range(300, 400)):
        model_save_name = os.path.join(dir2, f"{model}_sn_3.0_mod_seed_1_epoch_{epoch}.model")

        net.load_state_dict(torch.load(str(model_save_name), map_location=device), strict=True)
        net.eval()

        Xs = []
        ys = []
        for images, labels in train_loader:
            images = images.to(device)
            _ = net(images)
            embeddings = net.feature
            Xs.append(embeddings.cpu().detach().numpy())
            ys.append(labels.detach().numpy())
        X = np.concatenate(Xs)
        y = np.concatenate(ys)

        _, stats = normality_score(X, y, method)
        if stats < best_stats:
            best_stats = stats
            save_path = os.path.join(dir2, f"{model}_sn_3.0_mod_seed_1_best_gaussian_stats_{method}.model")
            torch.save(net.state_dict(), save_path)
            print("best gaussian model saved to ", save_path)

        # for images,_ in ood_test_loader:
        #     labels = np.ones(images.shape[0])*10 #标记label=10为OOD样本
        #     images = images.to(device)
        #     _ = net(images)
        #     embeddings = net.feature
        #     Xs.append(embeddings.cpu().detach().numpy())
        #     ys.append(labels)

        # X = np.concatenate(Xs)
        # y = np.concatenate(ys)
        # tsne = TSNE(n_components=2, init='pca', perplexity=50, random_state=0)
        # X_tsne = tsne.fit_transform(X)

        # fig = plot_embedding_2d(X_tsne, y, 10, f"epoch:{epoch}")
        # fig.savefig(os.path.join(dir2, f"stats_{epoch}.jpg"), dpi=300, bbox_inches='tight')
