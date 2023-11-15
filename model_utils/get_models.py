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
from . import resnet, vgg, vit ,swin

def get_model(arch, pretrained, num_classes, use_torchvision=True):
    if use_torchvision:#使用torchvision的官方实现
        print("call torchvision.models")
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
            if arch == "vgg16":
                lastlayer_dims = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(lastlayer_dims, num_classes)
            elif arch == "resnet50":
                lastlayer_dims = model.fc.in_features
                model.fc = nn.Linear(lastlayer_dims, num_classes)
            elif arch == "vit_b_16":
                lastlayer_dims = model.heads[0].in_features
                model.heads[0] = nn.Linear(lastlayer_dims, num_classes)
            elif arch == "swin_b":
                lastlayer_dims = model.heads[0].in_features
                model.heads[0] = nn.Linear(lastlayer_dims, num_classes)
        else:
            print("=> create model '{}'".format(arch))
            model = models.__dict__[arch](num_classes=num_classes)
    else: #使用专为cifar10 32x32实现的模型
        print("call specific models")
        if arch == "vgg16":
            model = vgg.VGG('VGG16')
        elif arch == "resnet50":
            model = resnet.ResNet50()
        elif arch == "vit_b_16":
            model = vit.ViT()
        elif arch =="swin_t":
            model = swin.swin_b()


    return model
