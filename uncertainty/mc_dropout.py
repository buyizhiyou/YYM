#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mc_dropout.py
@Time    :   2023/11/07 15:59:29
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import time
import sys
sys.path.append("../")

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

from utils.metircs import accuracy,mutual_info
from utils.visual import ProgressMeter, AverageMeter, Summary
from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model



def mc_dropout_predict(val_loader, model, device, N):
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

            start = time.time()
            outputs = []
            for _ in range(N):  # add mc_dropout
                prob = torch.softmax(model(images), axis=1)
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
    device = torch.device('cuda:1')
    model = get_model("vgg16", False, 10)
    model = model.to(device)
    model.eval()
    model.classifier[5].training = True  # 打开dropout
    checkpoint = torch.load("../saved_models/vgg16/vgg16_best_model.pth")
    model.load_state_dict(checkpoint['state_dict'])

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
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
    probs = mc_dropout_predict(val_loader, model, device, 8)

    mi = mutual_info(probs.cpu().numpy())


    
if __name__ == '__main__':
    main()
