#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mlp.py
@Time    :   2023/12/12 12:29:10
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import torch
from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, p, logvar, n_feature, n_hidden, n_output):
        super().__init__()
        self.p = p
        self.logvar = logvar
        self.hidden = nn.Linear(n_feature, n_hidden)  # hidden layer
        self.mid1 = nn.Linear(n_hidden, n_hidden)
        self.mid2 = nn.Linear(n_hidden, n_hidden)
        self.mid3 = nn.Linear(n_hidden, n_hidden)
        self.mid4 = nn.Linear(n_hidden, n_hidden)

        self.predict = nn.Linear(n_hidden, n_output)  # output layer mean
        self.get_var = nn.Linear(n_hidden, n_output)  # output layer var

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = F.relu(self.mid1(x))
        x = F.relu(self.mid2(x))
        x = F.relu(self.mid3(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.mid4(x))
        x = F.dropout(x, p=self.p, training=self.training)####注意！！！这里要使用self.training才可以通过model.eval()关闭dropout

        # get y and log_sigma
        y = self.predict(x)
        # For Epistemic-only model, we will disable the network arm that gives $\log\sigma$. the loss function should be a simple MSE
        if self.logvar:
            logvar = self.get_var(x)
        else:  # Epistemic only
            logvar = torch.zeros(y.size())
            
        return y, logvar
