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

from utils.metrics import accuracy,mutual_info,nll,ece,brier_score
from utils.visual import ProgressMeter, AverageMeter, Summary
from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model
from bayesian_torch.utils.util import predictive_entropy, mutual_information
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

def bnn_svi_predict(val_loader, model, device, num_monte_carlo=20):
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
            for _ in range(num_monte_carlo):# add bnn_svi
                prob  = torch.softmax(model.forward(images),dim=1)#输出的概率
                outputs.append(prob)
            outputs = torch.stack(outputs, dim=0)
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
    device = torch.device('cuda:1')
    model = get_model("resnet50", 10)
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type":  "Reparameterization",## Flipout or Reparameterization
        "moped_enable": "",  # initialize mu/sigma from the dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load("../saved_models/bayesian/resnet50/2023_11_17_20_53_02/resnet50_best_model_93.57.pth")
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
    
    dataname = "cifar10"
    if dataname=="mnist" or dataname=="fashionmnist":
        val_transform.transforms.insert(2,transforms.Lambda(lambda x:x.repeat(3,1,1)))
    _, val_dataset = get_dataset(dataname,"../data",None,val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
    bnn_svi_predict(val_loader, model, device, 20)


    
if __name__ == '__main__':
    main()
