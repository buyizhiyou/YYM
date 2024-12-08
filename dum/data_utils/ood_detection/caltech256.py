#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   caltech256.py
@Time    :   2024/11/28 10:52:33
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

'''
Caltech 256 是一个常用的计算机视觉数据集，主要用于物体识别任务，特别是对图像分类模型的评估。该数据集由加州理工学院（Caltech）开发，目标是通过一组多样化的物体类别来测试图像分类算法的泛化能力。
数据集概况：
类别数：256个类别，每个类别都有多张图像。
图像总数：约30,607张图像。
图像尺寸：图像尺寸变化较大，通常是自然图像，包含不同大小、角度、背景和光照条件。
图像内容：每个类别都包含特定的物体或场景，比如动物、交通工具、家用电器、植物、家具等。
类别介绍：
Caltech 256 数据集包含了256个物体类别，其中的一些示例类别包括：
鸟类（如“鸽子”、“鹦鹉”等）
动物（如“狗”、“猫”、“大象”等）
交通工具（如“汽车”、“火车”、“自行车”等）
家用物品（如“电视机”、“风扇”、“咖啡机”等）
自然物体（如“树木”、“花朵”等）
每个类别的样本数：
每个类别大约有 80 至 827 张图像（具体数量不同），所以每个类别的图像数量是相对不均匀的。
数据集的多样性非常丰富，包括了物体的不同视角、不同背景以及不同的拍摄条件。
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
    train_dataset = datasets.Caltech256(
        root=data_dir,
        split="train",
        transform=train_transform,
    )

    valid_dataset = datasets.Caltech256(
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
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize,
    ])

    data_dir = kwargs['root']
    dataset = datasets.Caltech256(root=data_dir, transform=transform, download=True)

    num_train = len(dataset)
    print(f"Caltech256 test:{num_train}")
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


# if __name__ == '__main__':
#     dataloader = get_test_loader(32, root="../../data")
#     for x in dataloader:
#         print(x[0].shape)
#         print(x[0].std())
#         # break
