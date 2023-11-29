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
import numpy as np 
sys.path.append("../")

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

from utils.metircs import accuracy,mutual_info,nll,ece,brier_score
from utils.visual import ProgressMeter, AverageMeter, Summary
from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model



def mc_dropout_predict(val_loader, model, device, num_monte_carlo=20):
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

            start = time.time()
            outputs = []
            for _ in range(num_monte_carlo):  # add mc_dropout
                prob = torch.softmax(model(images), axis=1)
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
    # num_classes = np.max(targets)+1
    # y_true = np.eye(num_classes)[targets]#转为One-hot
    # brier_score_ = brier_score(y_true, probs.cpu().detach().numpy())
    # ece_score = ece(targets, probs.cpu().detach().numpy())
    # nll_score = nll(y_true, probs.cpu().detach().numpy())
    # print(f"brier score:{brier_score_},ece score:{ece_score},nll score:{nll_score}")
    
    return probs, targets


def main():
    device = torch.device('cuda:1')
    model = get_model("vgg16", 10)
    model = model.to(device)
    model.eval()
    model.classifier[2].training = True  # 打开dropout
    checkpoint = torch.load("../saved_models/deterministic/vgg16/2023_11_24_15_25_21/vgg16_best_model_93.62.pth")
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


    _, val_dataset = get_dataset("svhn","../data", None,val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
    mc_dropout_predict(val_loader, model, device, 20)


    
if __name__ == '__main__':
    main()
