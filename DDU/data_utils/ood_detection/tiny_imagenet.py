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


def get_train_valid_loader( batch_size, augment, val_seed, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - val_seed: fix seed for reproducibility.
    - val_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg


    path = "/home/sq/YYM/extra/DDU/data/tiny-imagenet-200"
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomGrayscale(),  # add
        transforms.GaussianBlur(3),  # add
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # pytorch doc std
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.ImageFolder(train_path,
                                          transform=train_transform)

    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

    train_loader = torch.utils.data_utils.DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
    )
    valid_loader = torch.utils.data_utils.DataLoader(
        valid_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    path  = "/home/sq/YYM/extra/DDU/data/tiny-imagenet-200"
    val_path = os.path.join(path, "val")

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    val_loader = torch.utils.data_utils.DataLoader(
        val_dataset, batch_size=4, num_workers=1, pin_memory=pin_memory, shuffle=False,
    )

    return  val_loader