#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_dataset.py
@Time    :   2023/11/02 19:45:10
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar10_dataset(path):

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomGrayscale(),
            transforms.GaussianBlur(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=path, train=True, download=False, transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root=path, train=False, download=False, transform=val_transform
    )

    return train_dataset, val_dataset


def get_imagenet_dataset(path):
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset
