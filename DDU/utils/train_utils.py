"""
This module contains methods for training models.
"""

from enum import Enum
import torch
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from utils.eval_utils import accuracy
from utils.simclr_utils import ContrastiveLearningViewGenerator, get_simclr_pipeline_transform, info_nce_loss, supervisedContrastiveLoss
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
    log_interval = 50
    model.train()
    train_loss = 0
    num_samples = 0
    acc = 0

    if contrastive:
        activation = {}

        def get_activation(name):

            def hook(model, input, output):
                activation[name] = input[0]

            return hook

        # model.fc.register_forward_hook(get_activation('embedding'))
        model.projection_head.out.register_forward_hook(
            get_activation('embedding'))

    if label_smooth:#使用label smoothing，使特征空间更紧密
        loss_func = LabelSmoothing()
    else:
        loss_func = nn.CrossEntropyLoss()

    contrastiveGenerator = ContrastiveLearningViewGenerator(
        get_simclr_pipeline_transform(32))
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        labels = labels.to(device)
        batch_size = data.shape[0]
        optimizer.zero_grad()

        if contrastive:
            """
            类间对比loss
            """
            # logits = model(data)
            # loss1 = loss_func(logits, labels)
            # loss2 = supervisedContrastiveLoss(embeddings, labels, device,temperature=0.5)
            # loss = loss1 - 0.01 * loss2  #这个好一些？？
            # loss = loss1 + 0.01 * loss2
            """
            样本间对比loss
            """
            images = contrastiveGenerator(data)#对每个图片生成两个视角的增强
            images = torch.cat(images, dim=0)
            labels = torch.cat([labels,labels],dim=0)
            logits = model(images) #TODO:ADD projection head for model 使用resnet2

            embeddings = activation['embedding']
            logits2, labels2 = info_nce_loss(embeddings, batch_size, device)
            loss2 = F.cross_entropy(logits2, labels2)
            if (epoch<1000):#第一阶段，只训练对比loss
                loss = loss2
            else: #第二阶段，对比loss+分类loss
                loss1 = loss_func(logits, labels)
                loss = 100*loss1 + loss2
        else:
            logits = model(data)
            loss1 = loss_func(logits, labels)
            loss = loss1

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        acc1, acc5 = accuracy(logits, labels, (1, 5))
        acc += acc1.item() * len(data)
        if batch_idx % log_interval == 0:
            tqdm.write(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccu: {:.4f}"
                .format(epoch, batch_idx * len(data),
                        len(train_loader) * len(data),
                        100.0 * batch_idx / len(train_loader), loss.item(),
                        acc1.item()))

    tqdm.write(
        "====> Epoch: {} Average loss: {:.4f}\t Average Acc:{:.4f}".format(
            epoch, train_loss / num_samples, acc / num_samples))
    return train_loss / num_samples, acc / num_samples


def test_single_epoch(model, test_val_loader, device):
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
            acc1, acc5 = accuracy(logits, labels, (1, 5))

            acc += acc1.item() * len(data)
            num_samples += len(data)

    print("======> Test set ,accu:{:.4f}".format(acc / num_samples))
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
        return str(model_name) + strn + "seed_" + str(seed) + "_contrastive"
    else:
        return str(model_name) + strn + "seed_" + str(seed)
