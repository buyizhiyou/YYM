#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metircs.py
@Time    :   2023/11/02 20:54:16
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import numpy as np
from scipy import special
from scipy.stats import entropy

import torch
from torchmetrics.classification import MulticlassCalibrationError


def accuracy(output: torch.TensorType, target: torch.TensorType, topk: tuple = (1,)):
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
    

def predictive_entropy(probs: np.ndarray):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.(Aleatoric Uncertainty)

    Args:
        probs (np.ndarray): NxKxM array
    """
    return np.mean(entropy(probs, axis=2), axis=1)


def mutual_info(probs: np.ndarray):
    """Calculate mutual information (Epistemic Uncertainty)

    Args:
        probs (np.array): NxKxM array
    """
    assert isinstance(probs, np.ndarray), "probs should be np.array"
    assert len(probs.shape) == 3, f"probs shape {probs.shape} is wrong"
    return entropy(np.mean(probs, axis=1), axis=1)-np.mean(entropy(probs, axis=2), axis=1)

def expected_kl(probs: np.ndarray):
    """calculate expected kullback-leibler divergence (Epistemic Uncertainty)

    Args:
        probs (np.ndarray): NxKxM array
    """
    assert isinstance(probs, np.ndarray), "probs should be np.array"
    assert len(probs.shape) == 3, f"probs shape {probs.shape} is wrong"
    kl = entropy(np.mean(probs,axis=1)[...,None,:], probs+1e-30, axis=2) #+1e-30 to avoid divided by 0 and get inf  in KL divergence
    return np.mean(kl,axis=1)

def predictive_variance(probs: np.ndarray):
    """calculate predictive variance (Epistemic Uncertainty)

    Args:
        probs (np.ndarray):  NxKxM array

    Returns:
        _type_:  N array
    """
    assert isinstance(probs, np.ndarray), "probs should be np.array"
    assert len(probs.shape) == 3, f"probs shape {probs.shape} is wrong"
    variance = np.linalg.norm(probs-np.mean(probs,axis=1)[...,None,:],ord=2,axis=2)
    return np.sum(variance,axis=1)


def nll(y_true: np.ndarray, y_pred: np.ndarray):
    """
    calculate negative log likelihood
    """
    assert isinstance(y_pred, np.ndarray), "y_pred should be np.array"
    assert isinstance(y_true, np.ndarray), "y_true shold be np.array"
    return np.mean(-special.xlogy(y_true, y_pred))


def ece(targets: np.ndarray, y_pred: np.ndarray):
    """
    calculate Expected calibration error
    """
    assert isinstance(y_pred, np.ndarray), "y_pred should be np.array"
    assert isinstance(targets, np.ndarray), "targets shold be np.array"
    ece_score = MulticlassCalibrationError(num_classes=10,n_bins=15, norm='l2')(torch.Tensor(y_pred), torch.Tensor(targets))

    return ece_score.cpu().detach().numpy()



def brier_score(y_true:np.ndarray, y_pred: np.ndarray):
    """calculate brier score
    """
    assert isinstance(y_true, np.ndarray), "y_true shold be np.array"
    assert isinstance(y_pred, np.ndarray), "y_true shold be np.array"
    if len(y_pred.shape)==3:
        y_pred = np.mean(y_pred,axis=1)
    return np.mean(np.sum((y_pred - y_true)**2, axis=1))

#  scipy.stats.ranksums
def get_probability_interval(preds, interval=0.95):
    lower_int = (1. - interval) / 2
    upper_int = interval + lower_int

    # Calculate percentiles and mean
    lower = np.quantile(preds, lower_int, axis=0)
    mean = np.mean(preds, axis=0)
    upper = np.quantile(preds, upper_int, axis=0) 

    return lower, mean, upper

def picp(preds,y_test):
    lower, mean, upper = get_probability_interval(preds, 0.95)
    in_interval = sum([l <= y <= u for y, l, u in zip(y_test, lower, upper)])

    return in_interval/len(y_test)

def mpiw(preds,y_test):
    lower, mean, upper = get_probability_interval(preds, 0.95)

    return np.mean(upper-lower)

if __name__ == '__main__':
    arr = np.random.rand(10000,20,10)
    res = predictive_variance(arr)
    print(res.shape)