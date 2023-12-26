#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deep_ensemble.py
@Time    :   2023/11/07 17:06:56
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys

sys.path.append("../")
import time

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn

from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model
from utils.metrics import accuracy
from utils.visual import AverageMeter, ProgressMeter, Summary


class EnsembleModel(nn.Module):

    def __init__(self, model_name, models_path, device, num_classes=10):
        super().__init__()
        self.models_path = models_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.trained_models = self._load_weights()

    def _load_weights(self):
        trained_models = []
        for model_path in self.models_path:
            model = get_model(self.model_name, self.num_classes)
            model = model.to(self.device)
            model.eval()
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            trained_models.append(model)

        return trained_models

    def forward(self, x):
        outputs = []
        for model in self.trained_models:
            outputs.append(torch.softmax(model(x), axis=1))

        outputs = torch.stack(outputs, dim=0)  #NxBatchSizexnum_classes

        return torch.mean(outputs, axis=0)

    def mc_forward(self, x):
        outputs = []
        for model in self.trained_models:
            outputs.append(torch.softmax(model(x), axis=1))

        outputs = torch.stack(outputs, dim=0)  #NxBatchSizexnum_classes

        return outputs


def deep_ensembel_predict(val_loader, vgg_ensemble, device):
    inference_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [inference_time, top1],
                             prefix='Test: ')

    probs_list = []
    target_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            target_list.append(target)

            start = time.time()
            outputs = vgg_ensemble.mc_forward(images)

            probs_list.append(outputs)
            output = torch.mean(outputs, dim=0)

            # measure elapsed time
            inference_time.update(time.time() - start, images.size(0))
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            top1.update(acc1[0], images.size(0))

    progress.display_summary()
    # BatchesxNxBatchSizexNumClasses-->NxBatchSize*BatchesxNumClasses
    probs = torch.concat(probs_list, axis=1)
    # NxBatchSize*BatchesxNumClasses-->BatchSize*BatchesxNxNumClasses
    probs = torch.transpose(probs, 0, 1)
    targets = torch.concat(target_list, axis=0)

    return probs, targets


def main():
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    _, val_dataset = get_dataset("cifar10", "../data", None, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

    device = torch.device('cuda:3')
    models_path = [
        "../saved_models/deterministic/vgg16/2023_11_15_16_36_44/vgg16_best_model_91.78.pth",
        "../saved_models/deterministic/vgg16/2023_11_15_21_15_51/vgg16_best_model_92.90.pth",
    ]

    vgg_ensemble = EnsembleModel("vgg16", models_path, device)
    deep_ensembel_predict(val_loader, vgg_ensemble, device)


if __name__ == '__main__':
    main()
