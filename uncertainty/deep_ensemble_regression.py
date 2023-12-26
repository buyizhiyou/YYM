#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mc_dropout_regression.py
@Time    :   2023/12/12 16:13:51
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import sys
sys.path.append("../")
import numpy as np 
import torch 
from torch import nn 

from model_utils.mlp import MLPNet

class EnsembleRegressionModel(nn.Module):
    def __init__(self, models_path, device):
        super().__init__()
        self.models_path = models_path
        self.device = device
        self.trained_models = self._load_weights()

    def _load_weights(self):
        trained_models = []
        for model_path in self.models_path:
            model = MLPNet(0.5,True,1,32,1)
            model = model.to(self.device)
            model.eval()
            checkpoint = torch.load(model_path,map_location=self.device)
            model.load_state_dict(checkpoint)
            trained_models.append(model)

        return trained_models

    def forward(self, x):
        predictions, logvars = self.mc_forward(x)

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

    preds, logvars = network.mc_forward(x)

    preds = preds.cpu().detach().numpy()
    au = logvars.cpu().detach().numpy()
    au = np.sqrt(np.mean(np.exp(au), axis=0))#TODO: 先求平均还是后求平均
    pred_mean = np.mean(preds, axis=0)
    eu = np.sqrt(np.var(preds, axis=0))
    
    return preds.squeeze(), pred_mean.squeeze(), au.squeeze(), eu.squeeze()


if __name__ == '__main__':
    models_path = ["../saved_models/regression/deterministic/2023_12_20_20_53_00/mlp.pth",
                   "../saved_models/regression/deterministic/2023_12_20_20_53_00/mlp.pth"]
    device = "cuda:0"
    network = EnsembleRegressionModel(models_path,device)
    x = torch.randn(10,1).to(device)
    preds,preds_mean,au,eu = deep_ensemble_eval(network,x)

    import pdb;pdb.set_trace()