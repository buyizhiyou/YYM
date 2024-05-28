#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datasets_distance.py
@Time    :   2024/05/27 20:00:14
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

# Measuring Similarity Between Datasets' Features
#three metrics: the PAD, the MMD, and the Wasserstein distance.

import sys

sys.path.append("../")

import torch
import torch.nn as nn
from geomloss import SamplesLoss

import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.gauss as gauss
import data_utils.ood_detection.fer2013 as fer2013
import data_utils.ood_detection.dtd as dtd
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "fer2013": fer2013,
    "mnist": mnist,
    "lsun": lsun,
    "dtd": dtd,
    "gauss": gauss,
    "tiny_imagenet": tiny_imagenet
}


def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if self.fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul**(kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd_distance(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Maximum mean discrepancy (MMD) is a kernel based statistical test used to determine whether given two distribution
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    dist = torch.mean(XX + YY - XY - YX)

    return dist


if __name__ == '__main__':
    test_loader = dataset_loader["cifar10"].get_test_loader(root="../data", batch_size=1024, pin_memory=True)
    ood_name = ["dtd", "fer2013", "cifar100", "mnist", "lsun", "svhn", "tiny_imagenet", "gauss"]
    for data, _ in test_loader:
        data = data.reshape((data.shape[0], -1))
        break

    loss = SamplesLoss("sinkhorn", blur=0.5)

    for ood in ood_name:
        ood_test_loader = dataset_loader[ood].get_test_loader(root="../data", batch_size=1024, pin_memory=True)
        for ood_data, _ in ood_test_loader:
            ood_data = ood_data.reshape((ood_data.shape[0], -1))
            break

        assert data.shape == ood_data.shape, "shape conflicts..."

        # dist = mmd_distance(data, ood_data)
        dist = loss(data, ood_data)
        print(f"cifar10 vs {ood}:{dist.item():.4f}")
