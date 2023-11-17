#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bnn_svi.py
@Time    :   2023/11/17 15:27:29
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
from bayesian_torch.utils.util import predictive_entropy, mutual_information


def bnn_svi_predict(val_loader, model, device, num_monte_carlo):
    inference_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [inference_time,  top1],
        prefix='Test: ')

    probs_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            start = time.time()
            outputs = []
            for _ in range(num_monte_carlo):# add bnn_svi
                prob  = torch.softmax(model.forward(images),dim=1)#输出的概率
                outputs.append(prob)
            outputs = torch.stack(outputs, dim=0)
            probs_list.append(outputs)
            output = torch.mean(outputs, dim=0)

                        
            # predictive_uncertainty = predictive_entropy(outputs.data.cpu().numpy())
            # model_uncertainty = mutual_information(outputs.data.cpu().numpy())

            # measure elapsed time
            inference_time.update(time.time() - start, images.size(0))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))


    # BatchesxNxBatchSizexNumClasses-->NxBatchSize*BatchesxNumClasses
    probs = torch.concat(probs_list, axis=1)
    # NxBatchSize*BatchesxNumClasses-->BatchSize*BatchesxNxNumClasses
    probs = torch.transpose(probs, 0, 1)
    progress.display_summary()

    return probs


def main():
    device = torch.device('cuda:1')
    model = get_model("vgg16", False, 10, False)
    model = model.to(device)
    model.eval()
    # model.classifier[5].training = True  # 打开dropout
    model.classifier.training = True  # 打开dropout
    checkpoint = torch.load("../saved_models/vgg16/2023_11_15_16_36_44/vgg16_best_model_91.78.pth")
    model.load_state_dict(checkpoint['state_dict'])

    val_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
            ]
        )


    _, val_dataset = get_dataset("cifar10","../data", None,val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
    probs = bnn_svi_predict(val_loader, model, device, 8)

    mi = mutual_info(probs.cpu().numpy())


    
if __name__ == '__main__':
    main()
