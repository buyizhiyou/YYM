#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vgg.py
@Time    :   2023/11/15 14:32:15
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
""" vgg_bn for cifar10 32x32 image size"""

import torch
import torch.nn as nn
from net.spectral_normalization.spectral_norm_official import spectral_norm
import torch.nn.functional as F

from net.extra import ProjectionHead

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(
        self,
        vgg_name,
        num_classes=10,
        dropout=0.5,
        temp=1.0,
        coeff=3.0,
        spectral_normalization=True,
        mod=True,
    ):
        super(VGG, self).__init__()
        self.wrapped_conv = spectral_norm if spectral_normalization else nn.Identity()
        # self.wrapped_bn = spectral_norm if spectral_normalization else nn.Identity()
        self.wrapped_bn = nn.Identity()
        self.activation = nn.LeakyReLU(inplace=True) if mod else nn.ReLU(inplace=True)
        self.features = self._make_layers(cfg[vgg_name])

        self.fc_add = nn.Linear(512, 512)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(512, num_classes)

        self.projection_head = ProjectionHead(512, 256)
        self.feature = None

        self.temp = temp

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        out = self.fc_add(out)
        out = self.drop(self.activation(out))

        self.embedding = self.projection_head(out)  # 对比loss的embedding
        self.feature = out
        out = self.fc(out) / self.temp

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    self.wrapped_conv(nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                    self.wrapped_bn(nn.BatchNorm2d(x)), self.activation
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(spectral_normalization=True, mod=True, temp=1.0, **kwargs):
    model = VGG("VGG11", spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model


def vgg13(spectral_normalization=True, mod=True, temp=1.0, **kwargs):
    model = VGG("VGG13", spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model


def vgg16(spectral_normalization=True, mod=True, temp=1.0, **kwargs):
    model = VGG("VGG16", spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model
