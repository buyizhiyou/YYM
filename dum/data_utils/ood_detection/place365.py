#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   place365.py
@Time    :   2024/11/28 10:53:58
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''


'''
Place365 是一个用于场景分类的图像数据集，旨在帮助计算机视觉领域的研究人员开发和评估用于场景理解、图像分类和深度学习模型的算法。它是由哈佛大学和麻省理工学院（MIT）共同开发的，并广泛应用于测试图像分类模型，尤其是场景分类模型。

数据集概况：
类别数：365个场景类别。
图像总数：约1.8百万张图像。
图像来源：数据集中的图像来自于多种常见的图像来源，涵盖了日常生活中可能遇到的多种场景，比如城市街道、室内房间、自然风景等。
图像尺寸：每张图像的尺寸各异，通常为多种分辨率，且图像中包含了不同的光照条件、视角和拍摄距离。

类别介绍：
Place365 包含了 365 个场景类别，具体包括但不限于：
室内场景：如厨房、浴室、办公室、卧室、图书馆等。
城市景观：如街道、广场、城市天际线等。
自然景观：如森林、沙漠、湖泊、山脉等。
特定场景：如机场、博物馆、购物中心等。

数据集特点：
多样性：Place365 数据集的场景类别涵盖了广泛的日常生活场景，包含室内和室外各种不同的环境，极大地提高了数据集的多样性。
大规模数据：拥有超过 1.8 百万张图像，这使得 Place365 成为一个大规模的场景分类数据集，适合用于训练和评估深度学习模型。
高质量标注：每个图像都有对应的场景标签，标注准确，图像覆盖了各种环境和条件，适合场景分类任务。
挑战性：由于存在不同的场景在视觉上可能存在相似性（如一些室内场景可能非常相似），因此这对分类模型提出了较高的要求。

数据集的使用：
Place365 数据集常用于以下领域：
场景分类：主要用于测试和评估基于深度学习的图像分类模型，特别是卷积神经网络（CNN）等。
视觉理解与智能系统：适用于各种智能系统的训练，例如自动驾驶、机器人视觉、AR/VR 环境下的图像识别等。
迁移学习：由于其规模庞大和多样性，Place365 也常被用于迁移学习任务，作为预训练数据集来提升其他视觉任务的性能。

数据集特点：
类别不均衡：不同类别之间的样本数可能有所差异，部分场景类别可能包含更多图像，而其他类别则相对较少，这可能会影响训练模型的效果，尤其是在小样本类别上。
场景背景干扰：图像中的背景或拍摄条件可能对识别产生干扰，特别是一些类别之间视觉上的相似性较高（例如不同类型的城市景观），需要模型具备更强的区分能力。
现实环境中的挑战：图像的质量和光照条件存在差异，图像可能包括各种动态元素（如交通工具、行人等），这增加了分类任务的复杂性。
'''

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.0, num_workers=4, pin_memory=False, contrastive=0, **kwargs):
    
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize,
        ])

    # load the dataset
    data_dir = kwargs['root']
    train_dataset = datasets.Places365(
        root=data_dir,
        split="train",
        transform=train_transform,
    )

    valid_dataset = datasets.Places365(
        root=data_dir,
        split="train",
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    torch.manual_seed(1)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=6),
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = kwargs['root']
    dataset = datasets.Places365(
        root=data_dir,
        split="val",
        transform=transform,
        # target_transform=transforms.ToTensor(),
        download=True
    )

    num_train = len(dataset)
    print(f"places365 test:{num_train}")
    if (num_train >= 1000):
        indices = list(range(num_train))
        split = 1000
        np.random.seed(1)
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        dataset = Subset(dataset, valid_idx)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


if __name__ == '__main__':
    dataloader = get_test_loader(32, root="../../data")
    for x in dataloader:
        print(x[0].std())
        break
