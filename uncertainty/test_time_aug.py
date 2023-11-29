#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_time_aug.py
@Time    :   2023/11/07 17:11:21
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''


import sys
sys.path.append("../")
import time

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import ttach as tta
from torch import nn
from torchvision import transforms

from model_utils.get_models import get_model
from utils.metircs import accuracy,nll,ece,brier_score
from utils.visual import AverageMeter, ProgressMeter, Summary




def test_time_aug_predict(val_loader, model, device, tta_num=10):
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

            tta_transform = transforms.Compose(
                [
                    # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy(
                    #     'cifar10'), interpolation=transforms.InterpolationMode.BILINEAR),##效果不好
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
                ]
            )
            outputs = []
            start = time.time()
            for i in range(tta_num):
                images_ = tta_transform(images)
                prob = torch.softmax(model(images_), axis=1)
                outputs.append(prob)
            outputs = torch.stack(outputs, dim=0)
            probs_list.append(outputs)
            output = torch.mean(outputs, dim=0)

            # measure elapsed time
            inference_time.update(time.time() - start, images.size(0))
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            top1.update(acc1[0], images.size(0))

    # BatchesxNxBatchSizexNumClasses-->NxBatchSize*BatchesxNumClasses
    probs = torch.concat(probs_list, axis=1)
    # NxBatchSize*BatchesxNumClasses-->BatchSize*BatchesxNxNumClasses 
    probs = torch.transpose(probs, 0, 1)
    progress.display_summary()
    targets = torch.concat(target_list, axis=0)


    return probs,targets


def main():
    val_transform = transforms.Compose(
        [
            transforms.PILToTensor(), #返回torch.uint8
            transforms.Resize([32,32]),
        ]
    )
    val_dataset = datasets.CIFAR10(
        root="../data", train=False, download=False, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    device = torch.device('cuda:3')
    model = get_model("vgg16", 10)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(
        "../saved_models/deterministic/vgg16/2023_11_24_15_25_21/vgg16_best_model_93.62.pth")
    model.load_state_dict(checkpoint['state_dict'])
    test_time_aug_predict(val_loader, model, device)


if __name__ == '__main__':
    main()
