#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2023/11/16 15:02:20
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropyLoss(nn.Module):

    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AleatoricLoss(nn.Module):
    """
    Paper: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (8)
    
    """

    def __init__(self):
        super(AleatoricLoss, self).__init__()

    def forward(self, gt, pred_y, logvar):
        loss = torch.sum(0.5 * (torch.exp((-1) * logvar)) * (gt - pred_y)**2 +
                         0.5 * logvar)
        return loss
