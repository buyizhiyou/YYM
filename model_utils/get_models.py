#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_models.py
@Time    :   2023/11/07 16:15:16
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

from torch import nn
from torchvision import models

from . import resnet, swin, vgg, vit, resnet_variational


def get_model(arch,
              num_classes,
              use_torchvision=False,
              pretrained=False,
              use_bayesian=False):
    if use_torchvision:  # 使用torchvision的官方实现
        print("use torchvision official models...")
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
            if arch == "vgg16":
                lastlayer_dims = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(lastlayer_dims, num_classes)
            elif arch == "resnet50":
                lastlayer_dims = model.fc.in_features
                model.fc = nn.Linear(lastlayer_dims, num_classes)
            elif arch == "vit":
                lastlayer_dims = model.heads[0].in_features
                model.heads[0] = nn.Linear(lastlayer_dims, num_classes)
        else:
            print("=> create model '{}'".format(arch))
            if arch == "vgg16" or arch == "resnet50":
                model = models.__dict__[arch](num_classes=num_classes)
            elif arch == "vit":
                model = models.VisionTransformer(image_size=32,
                                                 patch_size=4,
                                                 num_layers=6,
                                                 num_heads=8,
                                                 hidden_dim=512,
                                                 mlp_dim=512,
                                                 dropout=0.,
                                                 attention_dropout=0.,
                                                 num_classes=num_classes)
    elif use_bayesian:  # TODO:添加更多bayesian模型
        print("use bayesian models...")
        if arch == "vgg16":
            pass
        elif arch == "resnet":
            model = resnet_variational.resnet20()
    else:  # 使用专为cifar10 32x32实现的模型
        print("use private models...")
        if arch == "vgg16":
            model = vgg.VGG('VGG16', num_classes)
        elif arch == "resnet50":
            model = resnet.ResNet50()
        elif arch == "vit":
            model = vit.ViT()
        elif arch == "swin_t":  # TODO:实验swin-t的效果
            model = swin.swin_b()

    return model
