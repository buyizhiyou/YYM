#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   misc.py
@Time    :   2023/11/07 14:37:01
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import os
import random

import numpy as np
import torch
from enum import Enum
from matplotlib import pyplot as plt


class argsdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_torch(seed: int = 2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置(速度会很慢)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name += "_aa"
    if args.label_smoothing:
        experiment_name += "_ls"
    if args.rcpaste:
        experiment_name += "_rc"
    if args.cutmix:
        experiment_name += "_cm"
    if args.mixup:
        experiment_name += "_mu"
    if args.off_cls_token:
        experiment_name += "_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name


def STD_FUNC(x): return 0.3 * (x + 2)  # 异方差函数
# MEAN_FUNC = np.cos
MEAN_FUNC  = lambda x: x^3
def gen_data(
    mean_fun=MEAN_FUNC,
    std_fun=STD_FUNC,
    std_const=0.1,
    hetero=False,
    occlude=False,
    aug=False,
    train_abs=5,
    test_abs=8,
    n_samples=2000,
    seed=1
):
    """
    mean_fun and std_fun: two normal function to
    engineer relatinship y = f_w(x) + sigma(x)
    # please keep std_fun larger than zero

    train/test abs are the absolute value of x ranges
    return two sets of tensor pairs
    """
    # test data as ground truth, 不加入噪声
    test_vec_size = int(n_samples)
    x_test = np.linspace(-test_abs, test_abs, test_vec_size)
    y_test = mean_fun(x_test)

    # train data with some problem ，加入一些噪声
    train_vec_size = int(n_samples)
    x_train = np.linspace(-train_abs, train_abs, train_vec_size)
    y_train = mean_fun(x_train)

    plot_title = "data:"
    if hetero:  # 添加异方差噪声
        plot_title = plot_title + " heteroskedastic"
        np.random.seed(seed)
        y_train = y_train + np.random.normal(0, np.abs(std_fun(x_train)), train_vec_size)
    else:  # 添加同方差噪声
        np.random.seed(seed)
        y_train = y_train + np.random.normal(0, std_const, train_vec_size)

    if occlude:
        plot_title = plot_title + " occlusion"
        indicies = np.arange(train_vec_size)
        n_piece = indicies // (train_vec_size / 5)
        new_indicies = np.concatenate(
            (indicies[n_piece != 1], indicies[n_piece == 1][1::6]), axis=None
        )  # 忽略掉一部分训练数据
        y_train = y_train[new_indicies]
        x_train = x_train[new_indicies]

    if aug:
        x_aug = np.linspace(-0.5, 0.5, 500)
        y_aug = mean_fun(x_aug)
        x_train = np.concatenate([x_train,x_aug])
        y_train = np.concatenate([y_train,y_aug])

    # plot
    plt.figure(figsize=(12, 4))
    plt.scatter(x_test, y_test, c="blue", s=7, label="test")
    plt.scatter(x_train, y_train,  s=7, alpha=0.3, label="train")
    plt.legend()
    plt.title(plot_title)

    # turn all array into tensors
    x_train = torch.from_numpy(x_train).float().view(-1, 1)
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    x_test = torch.from_numpy(x_test).float().view(-1, 1)
    y_test = torch.from_numpy(y_test).float().view(-1, 1)

    return (x_train, y_train, x_test, y_test)


