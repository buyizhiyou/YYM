#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_time_aug.py
@Time    :   2023/11/07 17:11:21
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''


import time
import sys
sys.path.append("../")

import torchvision.models as models
from torch import nn
import torch

from model_utils.get_models import get_model
from data_utils.get_dataset import get_cifar10_dataset
from utils.visual import ProgressMeter, AverageMeter, Summary
from utils.metircs import accuracy



def test_time_aug_predict(val_loader, model, device):
    inference_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [inference_time,  top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            end = time.time()
            #TODO: TTA 
            output = model(images)
            # measure elapsed time
            inference_time.update(time.time() - end, images.size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    progress.display_summary()


def main():
    _, val_dataset = get_cifar10_dataset("../data")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    device = torch.device('cuda:1')
    model = get_model("vgg16", False, 10)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load("../saved_models/vgg16/vgg16_best_model.pth")
    model.load_state_dict(checkpoint['state_dict'])
    test_time_aug_predict(val_loader, model, device)


if __name__ == '__main__':
    main()
