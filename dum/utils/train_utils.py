"""
This module contains methods for training models.
"""

import os
from enum import Enum
import random
import numpy as np

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.eval_utils import accuracy
from utils.loss import CenterLoss, LabelSmoothing, supervisedContrastiveLoss
from utils.simclr_utils import (ContrastiveLearningViewTransform, get_simclr_pipeline_transform, info_nce_loss)


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def seed_torch(seed: int = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置(速度会很慢)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train_single_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    aux_loss,
    optimizer_aux,
    device,
    contrastive=0,
    adv=0,
    label_smooth=False,
):
    """
    Util method for training a model for a single epoch.
    """
    model.train()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    num_samples = 0
    acc = 0
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    if contrastive==3 or contrastive==4:
        weight_center = 50  #TODO:后续在这里调整系数，逐步增大
        # weight_center = epoch*50/300
        
        
    if contrastive == 1 or contrastive == 2:
        activation = {}

        def get_activation1(name):

            def hook(model, input, output):
                activation[name] = input[0]

            return hook

        def get_activation2(name):

            def hook(model, input, output):
                activation[name] = output

            return hook

        model.projection_head.out.register_forward_hook(get_activation2('embedding'))

    if label_smooth:  #使用label smoothing，使特征空间更紧密
        ce_loss = LabelSmoothing()
    else:
        ce_loss = nn.CrossEntropyLoss()

    # for batch_idx, (x, y) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
    for batch_idx, (x, y) in enumerate((train_loader)):
        if (isinstance(x, list)):  #生成的多个视角的增强图片
            data = torch.cat(x, dim=0)
            labels = torch.cat([y, y], dim=0)
        else:
            data = x
            labels = y
        data = data.to(device)
        labels = labels.to(device)
        batch_size = data.shape[0]
        optimizer.zero_grad()

        if contrastive == 3 or contrastive == 4:
            optimizer_aux.zero_grad()
            
        if contrastive == 1:
            """
            类间对比loss
            """
            device2 = device  # 如果显存不够，可以将device2设为其他设备
            logits = model(data).to(device2)
            labels = labels.to(device2)
            embeddings = activation['embedding'].to(device2)
            loss1 = ce_loss(logits, labels)
            loss2 = supervisedContrastiveLoss(embeddings, labels, device2, temperature=0.1)
            # if (epoch < 300):  #第一阶段，前300epoch只训练对比loss
            #     loss = loss2
            # else:  #第二阶段，对比loss+分类loss
            #     loss1 = ce_loss(logits, labels)
            #     loss = 100 * loss1 + loss2
            #     # loss = loss1
            # loss = loss1 - 0.01 * loss2  #这个好一些？？让同一类尽量分散
            loss = loss1 + 1 * loss2  #让同一类尽量拥挤 #TODO:是距离选择有问题嘛？
            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)
        elif contrastive == 2:
            """
            样本间对比loss
            """
            logits = model(data)
            embeddings = activation['embedding']
            logits2, labels2 = info_nce_loss(embeddings, batch_size / 2, device)  #这里/2
            loss2 = F.cross_entropy(logits2, labels2)
            if (epoch < 300):  #第一阶段，只训练对比loss
                loss = loss2
            else:  #第二阶段，对比loss+分类loss
                loss1 = ce_loss(logits, labels)
                loss = 100 * loss1 + loss2

            acc1, _ = accuracy(logits2, labels2, (1, 5))
            acc += acc1.item() * len(data)
        elif contrastive == 3 or contrastive==4:  #centerloss或者修正的centerloss
            logits = model(data)
            embeddings = model.feature
            loss1 = ce_loss(logits, labels)
            loss2 = aux_loss(labels, embeddings)
            loss = loss1 + weight_center * loss2
            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)
        else:
            logits = model(data)
            loss1 = ce_loss(logits, labels)
            loss2 = torch.zeros(1)
            loss = loss1

            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)

        loss.backward()
        optimizer.step()
        if contrastive == 3 or contrastive == 4:
            for param in aux_loss.parameters():
                param.grad.data *= (1. / weight_center)
                
            # torch.nn.utils.clip_grad_norm_(aux_loss.parameters(), max_norm=1, norm_type=2)#gmmloss需要
            optimizer_aux.step()



        train_loss += loss.item() * len(data)
        train_loss1 += loss1.item() * len(data)
        train_loss2 += loss2.item() * len(data)
        num_samples += len(data)

    tqdm.write("====> Epoch: {} Average loss1: {:.4f} Average loss2: {:.4f}  Average loss: {:.4f}\t Average Acc:{:.4f}".format(
        epoch, train_loss1 / num_samples, train_loss2 / num_samples, train_loss / num_samples, acc / num_samples))
    return train_loss / num_samples, acc / num_samples


def test_single_epoch(epoch, model, test_val_loader, device):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    num_samples = 0
    acc = 0
    with torch.no_grad():
        for data, labels in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            acc1, _ = accuracy(logits, labels, (1, 5))

            acc += acc1.item() * len(data)
            num_samples += len(data)

    print("====>  Epoch: {}  Test set ,accu:{:.4f}".format(epoch, acc / num_samples))
    return acc / num_samples


def model_save_name(model_name, sn, mod, coeff, seed, contrastive=0):
    if sn:
        if mod:
            strn = "_sn_" + str(coeff) + "_mod_"
        else:
            strn = "_sn_" + str(coeff) + "_"
    else:
        if mod:
            strn = "_mod_"
        else:
            strn = "_"
    if contrastive:
        return str(model_name) + strn + "seed_" + str(seed) + f"_contrastive{contrastive}"
    else:
        return str(model_name) + strn + "seed_" + str(seed)


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
