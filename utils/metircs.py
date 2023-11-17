#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metircs.py
@Time    :   2023/11/02 20:54:16
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import numpy as np
import torch
from scipy import special
from scipy.stats import entropy
from torchmetrics.classification import BinaryCalibrationError


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
    assert isinstance(probs,np.ndarray), "probs should be np.array"
    return entropy(np.mean(probs, axis=1), axis=1)-np.mean(entropy(probs, axis=2), axis=1)


def nll(y_pred,y_true):
    """
    calculate negative log likelihood
    """
    assert isinstance(y_pred,np.ndarray), "y_pred should be np.array"
    assert isinstance(y_true,np.ndarray), "y_true shold be np.array"
    return np.mean(-special.xlogy(y_true, y_pred) - special.xlogy(1-y_true, 1-y_pred))


def ece(y_pred,y_true):
    """
    calculate calibration error
    """
    assert isinstance(y_pred,np.ndarray), "y_pred should be np.array"
    assert isinstance(y_true,np.ndarray), "y_true shold be np.array"
    ece_score =  BinaryCalibrationError(n_bins=2, norm='l2')(torch.Tensor(y_pred),torch.Tensor(y_true))

    return ece_score