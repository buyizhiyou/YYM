#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metircs.py
@Time    :   2023/11/02 20:54:16
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import torch
from scipy.stats import entropy
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mutual_info(probs):
    """calculate mutual information 

    Args:
        probs (np.array): NxKxM array
    """
    return entropy(np.mean(probs, axis=1), axis=1)-np.mean(entropy(probs, axis=2), axis=1)
