#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   extra.py
@Time    :   2024/04/12 11:19:29
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class ProjectionHead(nn.Module):

    def __init__(self, emb_size, head_size=512):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = F.normalize(h)
        h = self.hidden(h)
        h = F.relu(h)
        h = self.out(h)
        h = F.normalize(h)
        
        
        return h