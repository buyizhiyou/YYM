#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   laplace_approximation.py
@Time    :   2023/11/07 15:59:29
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys
import time

sys.path.append("../")

import numpy as np
import torch
import torch.distributions as dists
import torchvision.models as models
import torchvision.transforms as transforms
from laplace import Laplace
from laplace.curvature import AsdlGGN
from netcal.metrics import ECE
from torch import nn

from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model
from utils.metrics import accuracy, brier_score, ece, mutual_info, nll
from utils.visual import AverageMeter, ProgressMeter, Summary


def laplace_approx_predict(val_loader, la_model, device, num_mc_eval=20):
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
            # TODO: 预测准确率很低
            outputs = la_model.predictive_samples(images)
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

    return probs, targets


@torch.no_grad()
def predict(dataloader, model, device, laplace=False):
    probs = []
    targets = []

    for x, y in dataloader:
        if laplace:
            probs.append(model(x.to(device)))
        else:
            probs.append(torch.softmax(model(x.to(device)), dim=-1))

        targets.append(y)

    return torch.cat(probs).cpu().numpy(), torch.cat(targets).cpu().numpy()


def main():
    device = torch.device('cuda:1')
    model = get_model("vgg16", 10)
    model = model.to(device)
    checkpoint = torch.load(
        "../saved_models/deterministic/vgg16/2023_11_24_15_25_21/vgg16_best_model_93.62.pth",
        map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset, val_dataset = get_dataset("cifar10", "../data",
                                             val_transform, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True)

    # User-specified LA flavor
    la_model = Laplace(model,
                       'classification',
                       subset_of_weights='last_layer',
                       hessian_structure='kron',
                       backend=AsdlGGN)
    la_model.fit(train_loader)
    la_model.optimize_prior_precision(method='marglik', val_loader=val_loader)
    # User-specified predictive approx.
    probs, targets = laplace_approx_predict(val_loader,
                                            la_model,
                                            device,
                                            num_mc_eval=20)
    # import pdb;pdb.set_trace()

    probs_map, targets = predict(val_loader, model, device, laplace=False)
    acc_map = np.float32(probs_map.argmax(-1) == targets).mean()
    ece_map = ECE(bins=15).measure(probs_map, targets)
    nll_map = -dists.Categorical(torch.tensor(probs_map)).log_prob(
        torch.tensor(targets)).mean()

    print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}'
          )  # TODO:准确率略低于utils.metircs.accuracy计算出来的

    # Laplace
    la = Laplace(model,
                 'classification',
                 subset_of_weights='last_layer',
                 hessian_structure='kron')
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')

    probs_laplace, targets = predict(val_loader, la, device, laplace=True)
    acc_laplace = np.float32(probs_laplace.argmax(-1) == targets).mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace, targets)
    nll_laplace = -dists.Categorical(torch.tensor(probs_laplace)).log_prob(
        torch.tensor(targets)).mean()

    print(
        f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}'
    )


if __name__ == '__main__':
    main()
