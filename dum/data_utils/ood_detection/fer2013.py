#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fer2013.py
@Time    :   2024/05/28 10:59:04
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
"""
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered
so that the face is more or less centered and occupies about the same amount of space in each image. 
The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 
1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive,
for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image.
The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and
your task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. 
The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.0, num_workers=4, pin_memory=False, contrastive=0,size=32, **kwargs):
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
    train_dataset = datasets.FER2013(
        root=data_dir,
        split="train",
        transform=train_transform,
    )

    valid_dataset = datasets.FER2013(
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


def get_test_loader(batch_size, num_workers=4, pin_memory=False,size=32, **kwargs):
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
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    torch.manual_seed(1)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize,
    ])

    data_dir = kwargs['root']
    dataset = datasets.FER2013(
        root=data_dir,
        split="train",  #TODO: split="test"时,label=None，会报错
        transform=transform,
        # target_transform=transforms.ToTensor(),
    )

    num_train = len(dataset)
    print(f"fer2013 test:{num_train}")
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
