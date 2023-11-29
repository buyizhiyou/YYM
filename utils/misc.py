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


class argsdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_torch(seed:int=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置(速度会很慢)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
