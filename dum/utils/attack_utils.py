#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attack_utils.py
@Time    :   2024/05/24 11:06:09
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import sys

sys.path.append("../")

from .attack import PGD, FGSM, BIM, DeepFool, CW


def fgsm_attack(model, images, labels, device=None, epsilon=0.05, normalized=True):
    attack = FGSM(model, eps=epsilon)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def bim_attack(model, images, labels, device=None, epsilon=0.05, normalized=True):
    attack = BIM(model, eps=epsilon)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def deepfool_attack(model, images, labels, device=None, steps=50, normalized=True):
    attack = DeepFool(model, steps=steps)
    if device:
        attack.set_device(device)
    if normalized:  # If input images were normalized, then
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def pgd_attack(model, images, labels, device=None, epsilon=0.05, normalized=True):
    attack = PGD(model, eps=epsilon, alpha=10 / 255, steps=10, random_start=True)
    if device:
        attack.set_device(device)
    if normalized:
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images


def cw_attack(model, images, labels, device=None, normalized=True):
    attack = CW(model, c=1, kappa=0, steps=50, lr=0.01)
    if device:
        attack.set_device(device)
    if normalized:
        attack.set_normalization_used(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    adv_images = attack(images, labels)

    return adv_images
