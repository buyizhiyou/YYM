#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tiny_imagenet.py
@Time    :   2024/03/01 16:20:29
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import torch
import os
import numpy as np
from torch.utils.data import Subset

from torchvision import datasets
from torchvision import transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    data_dir = kwargs['root']
    train_path = os.path.join(data_dir, "tiny-imagenet-200", "train")
    val_path = os.path.join(data_dir, "tiny-imagenet-200", "val")

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomGrayscale(),  # add
        transforms.GaussianBlur(3),  # add
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # pytorch doc std
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)

    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

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

    data_dir = kwargs['root']
    val_path = os.path.join(data_dir, "tiny-imagenet-200", "val")

    torch.manual_seed(1)
    size = 224
    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(size, padding=4),
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.ImageFolder(val_path, transform=val_transform)
    num_train = len(dataset)
    print(f"tiny-imagenet test:{num_train}")
    if (num_train >= 1000):
        indices = list(range(num_train))
        split = 1000
        np.random.seed(1)
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        dataset = Subset(dataset, valid_idx)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return val_loader


if __name__ == '__main__':
    dataloader = get_test_loader(32, root="../../data")
    for x in dataloader:
        print(x[0].std())
        break
