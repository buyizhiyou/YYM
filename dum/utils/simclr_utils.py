#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simclr_utils.py
@Time    :   2024/03/06 20:35:08
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import transforms

np.random.seed(0)


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
        normalize,
    ])

    return data_transforms


class ContrastiveLearningViewTransform(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_loss(features, batch_size, device, temperature=0.7, n_views=2):
    #batch_size:256
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)  #torch.Size([512 ]) [0...127,0...127]
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  #torch.Size([512, 512],每一行兩個1
    labels = labels.to(device)  #每一行有两个1,分别是labels[i][i]和labels[i][i+256]，其中labels[i][i]是完全一样，lbales[i][i+256]是同张图片不同增强

    features = F.normalize(features, dim=1)  #将某一个维度除以那个维度对应的范数(默认是2范数

    similarity_matrix = torch.matmul(features, features.T)  #計算cos相似度

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)  #torch.Size([512, 511]),去除对角线
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  #torch.Size([512, 511])

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  #torch.Size([512, 1])
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  #torch.Size([512, 510])

    logits = torch.cat([positives, negatives], dim=1)  # put cos_sim of positive pairs as  first column  And cos_sim of negative pairs are the rest
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  #这里全赋值为0 crossEntropy = log_softmax+nllloss ,注意nllloss原理
    # NLLloss就是在做交叉熵损失函数的最后一步：预测结果的取负求和,
    # 它还顺带还帮你省了个OneHot编码，因为它是直接在 log(softmax(input)) 矩阵中，取出每个样本的target值对应的下标位置的预测结果进行取负求和运算
    # 这里labels全赋值0，就是说取每个样本的第0个值取负再求和

    logits = logits / temperature

    return logits, labels



