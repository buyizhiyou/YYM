#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bnn_svi_regression.py
@Time    :   2023/12/22 15:44:46
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''


import numpy as np


def bnn_svi_eval(network, X, mc_times=64):

    preds = []
    au = []
    network.eval()
    for t in range(mc_times):
        prediction, logvar = network(X)

        preds.append(prediction.data.cpu().detach().numpy())
        au.append(logvar.data.cpu().detach().numpy())

    preds = np.array(preds)
    au = np.array(au)
    au = np.sqrt(np.mean(np.exp(au), axis=0))
    pred_mean = np.mean(preds, axis=0)
    eu = np.sqrt(np.var(preds, axis=0))

    return preds.squeeze(), pred_mean.squeeze(), au.squeeze(), eu.squeeze()
