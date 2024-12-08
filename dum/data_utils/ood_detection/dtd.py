#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DTD.py
@Time    :   2024/05/28 10:59:04
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
"""
DTD is a texture database, consisting of 5640 images, organized according to a list of 47 terms (categories) 
inspired from human perception. There are 120 images for each category. Image sizes range between 300x300 and 640x640, 
and the images contain at least 90% of the surface representing the category attribute.
The images were collected from Google and Flickr by entering our proposed attributes and related terms as search queries.
The images were annotated using Amazon Mechanical Turk in several iterations. For each image we provide key attribute (main category) 
and a list of joint attributes.

The data is split in three equal parts, in train, validation and test, 40 images per class, for each split.
We provide the ground truth annotation for both key and joint attributes, as well as the 10 splits of the data we used for evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.0, num_workers=4, pin_memory=False, contrastive=0,size=32, **kwargs):
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
    train_dataset = datasets.DTD(root=data_dir, split="train", transform=train_transform, download=False)

    valid_dataset = datasets.DTD(root=data_dir, split="train", transform=valid_transform, download=False)

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
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = kwargs['root']
    dataset = datasets.DTD(
        root=data_dir,
        split="test",  #TODO: split="test"时,label=None，会报错
        transform=transform,
        download=False)

    num_train = len(dataset)
    print(f"dtd test:{num_train}")
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
