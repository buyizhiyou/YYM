#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   baseline_softmax.py
@Time    :   2023/11/07 19:25:16
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import time
import sys
sys.path.append("../")
import numpy as np 

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

from model_utils.get_models import get_model
from data_utils.get_datasets import get_dataset
from utils.visual import ProgressMeter, AverageMeter, Summary
from utils.metrics import accuracy,nll,ece,brier_score



def baseline_softmax_predict(val_loader, model, device):
    inference_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [inference_time,  top1],
        prefix='Test: ')

    probs_list = []
    target_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            target_list.append(target)

            end = time.time()
            output = model(images)
            probs_list.append(torch.softmax(output,axis=1))

            # measure elapsed time
            inference_time.update(time.time() - end, images.size(0))
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            top1.update(acc1[0], images.size(0))

    progress.display_summary()
    probs = torch.concat(probs_list, axis=0) #Sample_nums x Num_classes
    targets = torch.concat(target_list, axis=0)

    return probs ,targets


def main():
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ]
    )


    _, val_dataset = get_dataset("cifar","./data", None,val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    device = torch.device('cuda:2')
    model = get_model("vgg16", False, 10)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load("../saved_models/vgg16/vgg16_best_model.pth")
    model.load_state_dict(checkpoint['state_dict'])
    baseline_softmax_predict(val_loader, model, device)


if __name__ == '__main__':
    main()
