#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2023/11/15 14:31:38
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
""" resnet for cifar10 32x32 image size"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.spectral_normalization.spectral_norm_official import spectral_norm
from net.extra import ProjectionHead


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, wrapped_conv, wrapped_bn, activation, stride=1):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = wrapped_conv(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = wrapped_bn(nn.BatchNorm2d(planes))
        self.conv2 = wrapped_conv(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = wrapped_bn(nn.BatchNorm2d(planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(wrapped_conv(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                                          wrapped_bn(nn.BatchNorm2d(self.expansion * planes)))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, wrapped_conv, wrapped_bn, activation, stride=1):
        super(Bottleneck, self).__init__()

        self.activation = activation
        self.conv1 = wrapped_conv(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = wrapped_bn(nn.BatchNorm2d(planes))
        self.conv2 = wrapped_conv(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = wrapped_bn(nn.BatchNorm2d(planes))
        self.conv3 = wrapped_conv(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False))
        self.bn3 = wrapped_bn(nn.BatchNorm2d(self.expansion * planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(wrapped_conv(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                                          wrapped_bn(nn.BatchNorm2d(self.expansion * planes)))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, spectral_normalization=True, mod=True, temp=1.0, coeff=3.0, num_classes=10, dropout=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.wrapped_conv = spectral_norm if spectral_normalization else nn.Identity()
        # self.wrapped_bn = spectral_norm if spectral_normalization else nn.Identity()
        self.wrapped_bn = nn.Identity()
        self.activation = nn.LeakyReLU(inplace=True) if mod else nn.ReLU(inplace=True)
        self.conv1 = self.wrapped_conv(nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False))
        # self.conv1 = wrapped_conv(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False) #这是官方实现的版本
        self.bn1 = self.wrapped_bn(nn.BatchNorm2d(64))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc_add = nn.Linear(512 * block.expansion, 512 * block.expansion)  #添加一层线性层，为了dropout
        self.drop = nn.Dropout()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * block.expansion, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(512, num_classes),
        # )

        self.feature = None  # 这里抽出来倒数第二层feature，作为密度估计的高维特征
        self.embedding = None  # 对比loss的embedding
        self.temp = temp

        # add projection head for simclr
        self.projection_head = ProjectionHead(512 * block.expansion, 256)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.wrapped_conv, self.wrapped_bn, self.activation, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))  #torch.Size([1, 64, 32, 32])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.fc_add(out)
        out = self.drop(self.activation(out))

        self.embedding = self.projection_head(out)  # 对比loss的embedding
        self.feature = out
        out = self.fc(out) / self.temp

        return out


def resnet18(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model


def resnet50(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])

# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])

# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
