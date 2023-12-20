#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mc_dropout_regression.py
@Time    :   2023/12/12 16:13:51
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''


import numpy as np 
import torch 
from torch import nn 

from model_utils.mlp import MLPNet

class EnsembleRegressionModel(nn.Module):
    def __init__(self, models_path, device, num_classes=10):
        super().__init__()
        self.models_path = models_path
        self.num_classes = num_classes
        self.device = device
        self.trained_models = self._load_weights()

    def _load_weights(self):
        trained_models = []
        for model_path in self.models_path:
            model = MLPNet()
            model = model.to(self.device)
            model.eval()
            checkpoint = torch.load(model_path,map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            trained_models.append(model)

        return trained_models

    def forward(self, x):
        predictions = []
        logvars = []
        for model in self.trained_models:
            pred,logvar = model(x)
            predictions.append(pred)
            logvars.append(logvar)
        
        predictions = torch.stack(predictions, dim=0)
        logvars = torch.stack(logvars, dim=0)

        return torch.mean(predictions,axis=0),torch.mean(logvars,axis=0)


    def mc_forward(self, x):
        predictions = []
        logvars = []
        for model in self.trained_models:
            pred,logvar = model(x)
            predictions.append(pred)
            logvars.append(logvar)
        
        predictions = torch.stack(predictions, dim=0)
        logvars = torch.stack(logvars, dim=0)

        return predictions, logvars




def deep_ensemble_eval(network, x, dropout=True, mc_times=64):
    preds = []
    au = []

    predictions, logvars = network.mc_forward(x)
    prediction, logvar = network(x)

    preds.append(prediction.data.cpu().detach().numpy())
    au.append(logvar.data.cpu().detach().numpy())

    preds = np.array(preds)
    au = np.array(au)
    au = np.sqrt(np.mean(np.exp(au), axis=0))
    pred_mean = np.mean(preds, axis=0)
    eu = np.sqrt(np.var(preds, axis=0))
    
    return preds.squeeze(), pred_mean.squeeze(), au.squeeze(), eu.squeeze()