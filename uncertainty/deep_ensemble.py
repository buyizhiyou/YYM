#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deep_ensemble.py
@Time    :   2023/11/07 17:06:56
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import time
import sys
import numpy as np 
sys.path.append("../")
import gc

from torch import nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model_utils.get_models import get_model
from data_utils.get_datasets import get_dataset
from utils.visual import ProgressMeter, AverageMeter, Summary
from utils.metircs import accuracy

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)

def deep_ensembel_predict(val_loader,model_path, device):
    model = get_model("vgg16", False, 10)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
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
            prob = torch.softmax(model(images), axis=1)
            probs_list.append(prob.cpu().numpy())

            # measure elapsed time
            inference_time.update(time.time() - start, images.size(0))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(prob, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            if i>2:
                break

    # BatchesxNxBatchSizexNumClasses-->NxBatchSize*BatchesxNumClasses
    probs = np.concatenate(probs_list,axis=0)
    progress.display_summary()

    return probs


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

    #TODO:Ensembel train
    device = torch.device('cuda:2')
    models_path = ["../saved_models/vgg16/vgg16_best_model.pth",
                   "../saved_models/vgg16/2023_11_10_14_54_33/vgg16_best_model.pth",
                   "../saved_models/vgg16/2023_11_08_20_52_37/vgg16_best_model.pth"]
    outputs = []
    for model_path in models_path:
        probs = deep_ensembel_predict(val_loader ,model_path,  device)
        outputs.append(probs)
    outputs = torch.stack(outputs, dim=0)

if __name__ == '__main__':
    main()
