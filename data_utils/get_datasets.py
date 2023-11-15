#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_dataset.py
@Time    :   2023/11/02 19:45:10
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''
import os 
import torchvision.datasets as datasets

# train_transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x:x.repeat(3,1,1))
#     ]
# )

def get_dataset(name, path, train_transform, val_transform):
    if name == "cifar10":
        '''
        包含10类不同类型的32x32x3彩色图片，每类图片有6000张，共计60000张图片,50000张图片作为训练集，10000张作为测试集
        '''
        train_dataset = datasets.CIFAR10(
            root=path, train=True, download=False, transform=train_transform
        )

        val_dataset = datasets.CIFAR10(
            root=path, train=False, download=False, transform=val_transform
        )
    elif name == "cifar100":
        '''
        包含60000张32x32x3的彩色图像，涵盖100个类别,其中50000张图像用于训练，10000张图像用于测试
        '''
        train_dataset = datasets.CIFAR100(
            root=path, train=True, download=False, transform=train_transform
        )

        val_dataset = datasets.CIFAR100(
            root=path, train=False, download=False, transform=val_transform
        )
    elif name == "svhn":
        '''
        包含了73257张用于训练的图像、26032张用于测试的图像和531131张用于额外训练数据的额外图像。
        '''
        train_dataset = datasets.SVHN(
            root=path, split="train", download=True, transform=train_transform
        )

        val_dataset = datasets.SVHN(
            root=path, split="test", download=True, transform=val_transform
        )

    elif name == "mnist":
        '''
        由70000张手写数字0-9的灰度图像组成。其中，60000张用于训练，10000张用于测试。每张图像的大小为28×28像素
        '''
        train_dataset = datasets.MNIST(
            root=path, train=True, download=False, transform=train_transform
        )

        val_dataset = datasets.MNIST(
            root=path, train=False, download=False, transform=val_transform
        )

    elif name == "fashionmnist":
        '''
        由70000张服装灰度图像(训练集60000张，测试集10000张)组成,图片大小为28×28像素，代表10种不同类别的服装
        '''
        train_dataset = datasets.FashionMNIST(
            root=path, train=True, download=False, transform=train_transform
        )

        val_dataset = datasets.FashionMNIST(
            root=path, train=False, download=False, transform=val_transform
        )

    elif name == "imagenet":
        '''
        包含大约120万张训练图像，5万张验证图像和10万张测试图像,数据集中1000个类别
        '''
        train_path = os.path.join(path,"imageNet/train")
        val_path = os.path.join(path,"imageNet/val")

        train_dataset = datasets.ImageFolder(train_path,transform=train_transform)

        val_dataset = datasets.ImageFolder(val_path,transform=val_transform)

    else:
        raise Exception("参数错误")

    return train_dataset, val_dataset
