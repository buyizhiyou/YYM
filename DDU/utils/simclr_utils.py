#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simclr_utils.py
@Time    :   2024/03/06 20:35:08
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from torch.nn import functional as F

np.random.seed(0)


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size)),
    ])
    return data_transforms


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_loss(features, batch_size, device, temperature=0.7, n_views=2):
    #batch_size:256
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)],
                       dim=0)  #torch.Size([512 ])
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(
        device
    )  #每一行有两个1,分别是labels[i][i]和labels[i][i+256]，其中labels[i][i]是完全一样，lbales[i][i+256]是同张图片不同增强

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)  #torch.Size([512, 511])
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)  #torch.Size([512, 511])

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(
        labels.shape[0], -1)  #torch.Size([512, 1])
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)  #torch.Size([512, 510])

    logits = torch.cat(
        [positives, negatives], dim=1
    )  # put cos_sim of positive pairs as  first column  And cos_sim of negative pairs are the rest
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
        device)  #这里全赋值为0 crossEntropy = log_softmax+nllloss ,注意nllloss原理

    logits = logits / temperature

    return logits, labels


def supervisedContrastiveLoss(representations,
                              labels,
                              device,
                              temperature=0.7):
    """supervised contrastive loss

    Args:
        representations (_type_): batchsize*C
        labels (_type_): batchsize*1
        temperature (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: scalar
    """

    T = temperature  #温度参数T
    n = labels.shape[0]  # batch
    #这步得到它的相似度矩阵
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                            representations.unsqueeze(0),
                                            dim=2)
    #这步得到它的labels矩阵，相同labels的位置为1
    mask = torch.ones_like(similarity_matrix).to(device) * (labels.expand(
        n, n).eq(labels.expand(n, n).t()))
    #这步得到它的不同类的矩阵，不同类的位置为1
    mask_no_sim = torch.ones_like(mask).to(device) - mask
    #这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_diag_0 = torch.ones(n, n).to(device) - torch.eye(n, n).to(device)
    #这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix / T)
    #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix * mask_diag_0
    #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask * similarity_matrix
    #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
    no_sim = similarity_matrix - sim
    #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
    no_sim_sum = torch.sum(no_sim, dim=1)
    '''
    将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
    至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
    每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
    '''
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
    '''
    由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
    全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
    '''
    loss = mask_no_sim + loss + torch.eye(n, n).to(device)
    #接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  #求-log
    loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)  #将所有数据都加起来除以2n

    return loss
