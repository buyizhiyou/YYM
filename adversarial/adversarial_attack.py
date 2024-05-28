#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   adversarial_attack.py
@Time    :   2023/11/22 16:09:35
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model


def fgsm_attack(model,
                images,
                labels,
                epsilon=8 / 255,
                device=None,
                normalized=True):
    attack = torchattacks.FGSM(model, eps=epsilon)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def bim_attack(model,
               images,
               labels,
               epsilon=8 / 255,
               device=None,
               normalized=True):
    attack = torchattacks.BIM(model, eps=epsilon)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def deepfool_attack(model,
                    images,
                    labels,
                    steps=50,
                    device=None,
                    normalized=True):
    attack = torchattacks.DeepFool(model, steps=steps)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def pgd_attack(model,
               images,
               labels,
               epsilon=8 / 255,
               device=None,
               normalized=True):
    attack = torchattacks.PGD(model,
                              eps=epsilon,
                              alpha=1 / 255,
                              steps=10,
                              random_start=True)
    if device:
        attack.set_device(device)
    if normalized:
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def cw_attack(model, images, labels, device=None, normalized=True):
    attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    if device:
        attack.set_device(device)
    if normalized:
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def onepixel_attack(model, images, labels, normalized):
    attack = torchattacks.OnePixel(model,
                                   pixels=1,
                                   steps=10,
                                   popsize=10,
                                   inf_batch=128)
    if normalized:
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465),
                                      std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def test_adv(model: nn.Module,
             device: str,
             test_loader: DataLoader,
             epsilon: float = 0.01,
             method: str = "fgsm"):
    correct = 0  # Accuracy counter
    adv_examples = []
    model = model.to(device)
    for images, target in test_loader:
        images, target = images.to(device), target.to(device)
        init_prob = torch.softmax(model(images), axis=1)
        clean_prob, init_pred = init_prob.max(1, keepdim=True)
        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # 攻击发生在原始图片
        if method == "fgsm":
            perturbed_images = fgsm_attack(model, images, target, epsilon)
        elif method == "bim":
            perturbed_images = bim_attack(model, images, target, epsilon)
        elif method == "pgd":
            perturbed_images = pgd_attack(model, images, target, epsilon)
        elif method == "cw":
            perturbed_images = cw_attack(model, images, target, epsilon)
        elif method == "onepixel":
            perturbed_images = onepixel_attack(model, images, target, epsilon)

        final_prob = torch.softmax(model(perturbed_images), axis=1)
        adv_prob, final_pred = final_prob.max(1, keepdim=True)

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_examples.append(
                    (images.cpu().detach().numpy().squeeze(0),
                     perturbed_images.cpu().detach().numpy().squeeze(0),
                     init_pred.item(), final_pred.item(), clean_prob.item(),
                     adv_prob.item()))
        else:
            if len(adv_examples
                   ) < 5:  # Save 5 adv examples for visualization later
                adv_examples.append(
                    (images.cpu().detach().numpy().squeeze(0),
                     perturbed_images.cpu().detach().numpy().squeeze(0),
                     init_pred.item(), final_pred.item(), clean_prob.item(),
                     adv_prob.item()))

    final_acc = correct / float(len(test_loader))
    print(
        f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}"
    )

    return final_acc, adv_examples


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = get_model("vgg16", 10, use_torchvision=True)
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(
        "../saved_models/deterministic/vgg16/2023_11_14_15_54_52/vgg16_best_model_92.14.pth"
    )
    model.load_state_dict(checkpoint['state_dict'])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    _, val_dataset = get_dataset("cifar10", "../data", None, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test_adv(model, device, val_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
