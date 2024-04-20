#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2023/11/16 15:02:20
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import torch
import torch.nn as nn
from torch.autograd.function import Function


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, global_feat, labels, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CenterLoss(nn.Module):
    """https://github.com/jxgu1016/MNIST_center_loss_pytorch/tree/master

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


def supervisedContrastiveLoss(representations, labels, device, temperature=0.5):
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
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # N*N
    #这步得到它的labels矩阵，相同labels的位置为1
    mask = torch.ones_like(similarity_matrix).to(device) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
    #这步得到它的不同类的矩阵，不同类的位置为1
    mask_not_eq = torch.ones_like(mask).to(device) - mask
    #这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_no_diag = torch.ones(n, n).to(device) - torch.eye(n, n).to(device)
    #这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix / T)
    #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix * mask_no_diag
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
    loss = torch.div(sim, sim_sum)  #N*N
    '''
    由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
    全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
    '''
    loss = mask_not_eq + loss + torch.eye(n, n).to(device)
    #接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  #求-log
    # loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)  #将所有数据都加起来除以2n
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss


if __name__ == '__main__':
    x = torch.randn(7, 1024).to("cuda:0")
    y = torch.tensor([1, 2, 3, 1, 2, 3, 1]).to("cuda:0")
    supervisedContrastiveLoss(x, y, "cuda:0")