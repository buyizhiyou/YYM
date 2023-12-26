#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mc_dropout_regression.py
@Time    :   2023/12/12 16:13:51
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import numpy as np


def mc_dropout_eval(network, X, dropout=True, mc_times=64):
    preds = []
    au = []

    for t in range(mc_times):
        # For Aleatoric-only model, we assume no variance in predictives. Therefore we will disable dropout in both training and testing time.
        if dropout:  # 打开epistemic uncertainty计算
            prediction, logvar = network(X)
        else:
            network.eval()  # 关闭test dropout, 只计算aleatoric uncertainty
            prediction, logvar = network(X)

        preds.append(prediction.data.cpu().detach().numpy())
        au.append(logvar.data.cpu().detach().numpy())

    preds = np.array(preds)
    au = np.array(au)
    au = np.sqrt(np.mean(np.exp(au), axis=0))
    pred_mean = np.mean(preds, axis=0)
    eu = np.sqrt(np.var(preds, axis=0))

    return preds.squeeze(), pred_mean.squeeze(), au.squeeze(), eu.squeeze()
