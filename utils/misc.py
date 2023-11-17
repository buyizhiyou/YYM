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
    #cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

