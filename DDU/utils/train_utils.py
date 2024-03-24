"""
This module contains methods for training models.
"""

import os
import yaml
from enum import Enum
import torch
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from utils.eval_utils import accuracy
from utils.simclr_utils import ContrastiveLearningViewTransform, get_simclr_pipeline_transform, info_nce_loss, supervisedContrastiveLoss
from utils.loss import LabelSmoothing
from torchvision.transforms import transforms

import torch
import torch.nn.functional as F


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       contrastive,
                       label_smooth=False,
                       loss_mean=False):
    """
    Util method for training a model for a single epoch.
    """
    model.train()
    train_loss = 0
    num_samples = 0
    acc = 0

    if contrastive:
        activation = {}

        def get_activation1(name):
            def hook(model, input, output):
                activation[name] = input[0]
            return hook
        
        def get_activation2(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        if contrastive == 1:
            model.fc.register_forward_hook(get_activation1('embedding'))
        elif contrastive ==2:
            model.projection_head.out.register_forward_hook(
                get_activation2('embedding'))

    if label_smooth:  #使用label smoothing，使特征空间更紧密
        loss_func = LabelSmoothing()
    else:
        loss_func = nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        if (type(x) == list):  #生成的多个视角的增强图片
            data = torch.cat(x, dim=0)
            labels = torch.cat([y, y], dim=0)
        else:
            data = x
            labels = y 
        data = data.to(device)
        labels = labels.to(device)
        batch_size = data.shape[0]
        optimizer.zero_grad()
        logits = model(data)  #TODO:ADD projection head for model 使用resnet2
        if contrastive == 1:
            """
            类间对比loss
            """
            embeddings = activation['embedding']
            loss1 = loss_func(logits, labels)
            loss2 = supervisedContrastiveLoss(embeddings,
                                              labels,
                                              device,
                                              temperature=0.5)
            # if(epoch):第一阶段,只训练对比loss
            loss = loss1 - 0.01 * loss2  #这个好一些？？
            # loss = loss1 + 0.01 * loss2
            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)
        elif contrastive == 2:
            """
            样本间对比loss
            """
            embeddings = activation['embedding']
            logits2, labels2 = info_nce_loss(embeddings, batch_size / 2,
                                             device)  #这里/2
            loss2 = F.cross_entropy(logits2, labels2)
            if (epoch < 300):  #第一阶段，只训练对比loss
                loss = loss2
            else:  #第二阶段，对比loss+分类loss
                loss1 = loss_func(logits, labels)
                loss = 100 * loss1 + loss2

            acc1, _ = accuracy(logits2, labels2, (1, 5))
            acc += acc1.item() * len(data)
        else:
            loss1 = loss_func(logits, labels)
            loss = loss1

            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)


        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)



    tqdm.write(
        "====> Epoch: {}  Average loss: {:.4f}\t Average Acc:{:.4f}".format(
            epoch, train_loss / num_samples, acc / num_samples))
    return train_loss / num_samples, acc / num_samples


def test_single_epoch(epoch, model, test_val_loader, device):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    loss = 0
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

    print("====>  Epoch: {}  Test set ,accu:{:.4f}".format(
        epoch, acc / num_samples))
    return acc / num_samples


def model_save_name(model_name, sn, mod, coeff, seed, contrastive):
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
        return str(model_name) + strn + "seed_" + str(
            seed) + f"_contrastive{contrastive}"
    else:
        return str(model_name) + strn + "seed_" + str(seed)


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'),
              'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
