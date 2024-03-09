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
from utils.simclr_utils import ContrastiveLearningViewGenerator, get_simclr_pipeline_transform, info_nce_loss



import torch
import torch.nn.functional as F


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       contrastive,
                       loss_function="cross_entropy",
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
        model.projection_head.out.register_forward_hook(get_activation('embedding'))

    contrastiveGenerator = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32))
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss1 = F.cross_entropy(logits, labels)
        if contrastive:
            # loss2 = supervisedContrastiveLoss(embeddings, labels, device)
            # loss = loss1 - 0.1 * loss2
            images = contrastiveGenerator(data)
            images = torch.cat(images, dim=0)
            model(images)  #TODO:ADD projection head for model
            embeddings = activation['embedding']
            logits2, labels2 = info_nce_loss(embeddings)
            loss2 = F.cross_entropy(logits2, labels2)
            loss = loss1 + 0.1 * loss2
        else:
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
