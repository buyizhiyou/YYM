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
from utils.simclr_utils import ContrastiveLearningViewTransform, get_simclr_pipeline_transform, info_nce_loss
from utils.loss import supervisedContrastiveLoss, LabelSmoothing, CenterLoss
from torchvision.transforms import transforms


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def train_single_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    criterion_center,
    optimizer_centloss,
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
    weight_center = 0.01   


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

        if contrastive == 3:  #centerloss
            # model.fc.register_forward_hook(get_activation1('embedding'))
            pass
            # centerloss = CenterLoss(10, model.fc.in_features)
        else:  #ConLoss or supConLoss
            model.projection_head.out.register_forward_hook(get_activation2('embedding'))

    if label_smooth:  #使用label smoothing，使特征空间更紧密
        loss_func = LabelSmoothing()
    else:
        loss_func = nn.CrossEntropyLoss()

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
        optimizer_centloss.zero_grad()

        if contrastive == 1:
            """
            类间对比loss
            """
            device2 = device  # 如果显存不够，可以将device2设为其他设备
            logits = model(data).to(device2)
            labels = labels.to(device2)
            embeddings = activation['embedding'].to(device2)
            loss1 = loss_func(logits, labels)
            loss2 = supervisedContrastiveLoss(embeddings, labels, device2, temperature=0.1)
            # if (epoch < 300):  #第一阶段，前300epoch只训练对比loss
            #     loss = loss2
            # else:  #第二阶段，对比loss+分类loss
            #     loss1 = loss_func(logits, labels)
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
                loss1 = loss_func(logits, labels)
                loss = 100 * loss1 + loss2

            acc1, _ = accuracy(logits2, labels2, (1, 5))
            acc += acc1.item() * len(data)
        elif contrastive == 3:
            logits = model(data)
            # embeddings = activation['embedding']
            embeddings = model.feature
            loss1 = loss_func(logits, labels)
            loss2 = criterion_center(labels, embeddings)
            loss = loss1 + weight_center * loss2

            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)
        elif adv == 1:
            """对抗训练"""
            if batch_idx % 20 == 0:
                data.requires_grad = True  #data.required_grad区分,用required_grad梯度为None
                logits = model(data)
                loss = loss_func(logits, labels)

                model.zero_grad()
                loss.backward()

                # Collect ``datagrad``
                data_grad = data.grad.data
                data_denorm = data * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
                perturbed_data = fgsm_attack(data_denorm, 0.01, data_grad)
                perturbed_data_normalized = (perturbed_data - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

                data2 = torch.concat([data, perturbed_data_normalized])
                labels2 = torch.concat([labels, labels])
                logits2 = model(data2)
            else:
                logits2 = model(data)
                labels2 = labels
            loss = loss_func(logits2, labels2)
            acc1, _ = accuracy(logits2, labels2, (1, 5))
            acc += acc1.item() * len(data)
        else:
            logits = model(data)
            loss1 = loss_func(logits, labels)
            loss = loss1

            acc1, _ = accuracy(logits, labels, (1, 5))
            acc += acc1.item() * len(data)

        loss.backward()
        optimizer.step()
        for param in criterion_center.parameters():
            param.grad.data *= (1. / weight_center)
        optimizer_centloss.step()

        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
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
