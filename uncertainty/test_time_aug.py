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

from torch import nn
import torch
from torchvision import transforms 
import torchvision.models as models
import torchvision.datasets as datasets

from model_utils.get_models import get_model
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

    probs_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            tta_transform = transforms.Compose(
                [
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy('cifar10'),interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
                ]
            )
            outputs = []
            start = time.time()
            for i in range(8):
                images_ =  tta_transform(images)
                prob = torch.softmax(model(images_), axis=1)
                outputs.append(prob)
            outputs = torch.stack(outputs, dim=0)
            probs_list.append(outputs)
            output = torch.mean(outputs, dim=0)

            # measure elapsed time
            inference_time.update(time.time() - start, images.size(0))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


    # BatchesxNxBatchSizexNumClasses-->NxBatchSize*BatchesxNumClasses
    probs = torch.concat(probs_list, axis=1)
    # NxBatchSize*BatchesxNumClasses-->BatchSize*BatchesxNxNumClasses
    probs = torch.transpose(probs, 0, 1)
    progress.display_summary()

    return probs

def main():
    val_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.PILToTensor()
        ]
    )
    val_dataset = datasets.CIFAR10(
        root="../data", train=False, download=False, transform=val_transform)
    

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    device = torch.device('cuda:3')
    model = get_model("vgg16", False, 10)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load("../saved_models/vgg16/2023_11_14_15_54_52/vgg16_best_model_92.13999938964844.pth")
    model.load_state_dict(checkpoint['state_dict'])
    test_time_aug_predict(val_loader, model, device)


if __name__ == '__main__':
    main()
