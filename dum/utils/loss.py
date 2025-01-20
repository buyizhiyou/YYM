#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2023/11/16 15:02:20
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys
import torch
import torch.nn as nn
from pytorch_metric_learning import losses  # 这个库里实现了很多metric learning的loss
from torch.autograd.function import Function
from torch.nn import functional as F

import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        """Constructor for the LabelSmoothing module.
        smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
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

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-30, 1, 1)]
class GMMRegularizationLoss(nn.Module):

    def __init__(self, num_classes, feature_dim, device, lambda_compact=1.0, lambda_separate=1.0, lambda_gmm=1.0, epsilon=1e-5):
        """
        初始化 GMM 辅助损失的参数
        num_classes: 类别数
        feature_dim: 特征维度
        lambda_compact: 类内紧密性损失的权重
        lambda_separate: 类间可分性损失的权重
        lambda_gmm: GMM 对数似然损失的权重
        epsilon: 平滑项，防止数值不稳定
        """
        super(GMMRegularizationLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_compact = lambda_compact
        self.lambda_separate = lambda_separate
        self.lambda_gmm = lambda_gmm
        self.epsilon = epsilon
        self.device = device

        # 初始化高斯分布的中心 (mu) 和协方差矩阵 (sigma)
        self.mu = nn.Parameter(torch.randn(num_classes, feature_dim, device=device))  # 直接初始化在目标设备上
        # self.sigma = nn.Parameter(torch.eye(feature_dim, device=device).repeat(num_classes, 1, 1))  # 同上
        init_matrix = torch.randn(num_classes, feature_dim, feature_dim, device=device)
        init_matrix = torch.tril(init_matrix)  # 强制为下三角矩阵
        diag_indices = torch.arange(feature_dim)  # 对角线索引
        init_matrix[:, diag_indices, diag_indices] = F.softplus(init_matrix[:, diag_indices, diag_indices])
        self.cholesky_factors = nn.Parameter(init_matrix)  # 为了保证协方差矩阵的正定性，这里学习cholesky分解的参数

    def forward(self, labels, features):
        """
        计算总损失
        features: 特征张量，形状为 [batch_size, feature_dim]
        labels: 标签张量，形状为 [batch_size]
        """
        # 1. 计算类内紧密性损失 (compactness)
        compact_loss = self.compute_compact_loss(features, labels)
        # 2. 计算类间可分性损失 (separability)
        separate_loss = self.compute_separate_loss()
        # 3. 计算 GMM 对数似然损失
        gmm_loss = self.compute_gmm_loss(features, labels)
        #4. 计算一个正则化项，强制 cholesky_factors 保持较小
        # lambda_reg = 1 # 正则化的超参数
        # reg_loss = lambda_reg * torch.norm(self.cholesky_factors, p='fro')

        # 总损失
        # total_loss = compact_loss - separate_loss - gmm_loss + reg_loss
        total_loss = compact_loss - separate_loss - gmm_loss

        return total_loss

    def compute_compact_loss(self, features, labels):
        """
        计算类内紧密性损失
        """
        # 获取每个样本对应的类别中心
        mu_y = self.mu[labels]  # 形状 [batch_size, feature_dim]
        # 计算每个样本与类别中心的欧氏距离平方
        distances = torch.sum((features - mu_y)**2, dim=1)  # [batch_size]
        return distances.mean()

    def compute_separate_loss(self):
        """
        计算类间可分性损失
        """
        # 计算每对类别中心之间的距离平方
        mu_diff = self.mu.unsqueeze(1) - self.mu.unsqueeze(0)  # [num_classes, num_classes, feature_dim]
        distances = torch.sum(mu_diff**2, dim=-1) + self.epsilon  # [num_classes, num_classes]
        # 只考虑不同类别之间的距离
        mask = torch.eye(self.num_classes, device=self.mu.device).bool()
        inter_class_distances = distances[~mask].view(self.num_classes, self.num_classes - 1)
        separability = inter_class_distances

        return separability.mean()

    def compute_gmm_loss(self, features, labels):
        """
        计算 GMM 对数似然损失
        """
        # 获取所有类别的均值和协方差矩阵
        mu = self.mu
        L = self.cholesky_factors


        # 计算协方差矩阵
        sigma = torch.matmul(L, L.transpose(-1, -2))
        for jitter_eps in JITTERS:
            try:  #协方差矩阵要求正定,但是样本协方差矩阵不一定正定,所以加上对角阵转化为正定的
                jitter = jitter_eps * torch.eye(
                    self.feature_dim,
                    device=self.device,
                ).unsqueeze(0)
                mvn = MultivariateNormal(mu, covariance_matrix=(sigma+jitter))
            except:
                continue

        # 批量计算对数似然
        features_expanded = features.unsqueeze(1)
        try:
            log_likelihood = mvn.log_prob(features_expanded) 
        except:
            import pdb;pdb.set_trace()
            sys.exit()

        log_likelihood = torch.gather(log_likelihood, dim=1, index=labels.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)


        return log_likelihood.mean()


class CenterLoss(nn.Module):
    """Center loss
    """

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        self.device = device

    def forward_bk(self, labels, x):
        """
        centerloss
        """

        # center = self.centers[labels]
        # dist = (x-center).pow(2).sum(dim=-1)
        # loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        # 下面的代码计算类内距离（与中心的距离）
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        #(x-c)^2=x^2+c^2-2*x*c.t()

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        intra_class_distance = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        # # 下面的代码计算类间距离
        # num_classes = self.num_classes
        # center_dists = torch.cdist(self.centers, self.centers, p=2)  # 计算所有类别中心之间的欧氏距离
        # # 排除自身距离（对角线元素为 0）
        # mask = torch.eye(num_classes, device=center_dists.device).bool()
        # center_dists = center_dists.masked_fill(mask, 0)
        # # 计算类间距离的平均值
        # inter_class_distance = center_dists.sum() / (num_classes * (num_classes - 1))
        # # print(f"Inra-class Distance (Average): {intra_class_distance.item()}",f"Inter-class Distance (Average): {inter_class_distance.item()}")
        # loss = intra_class_distance/(10*inter_class_distance+1e-12)

        loss = intra_class_distance

        return loss

    def forward(self, labels, x, delta=1e-6):
        batch_size = x.size(0)
        num_classes = self.centers.size(0)

        # 计算每个样本到其所属类别中心的距离 ||x_i - c_{y_i}||^2
        mask = F.one_hot(labels, num_classes).float()
        dist_to_own_center = ((x - self.centers[labels])**2).sum(dim=1)

        # 计算类别中心之间的距离 ||c_{y_i} - c_j||^2
        expanded_own_centers = self.centers[labels].unsqueeze(1).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        expanded_centers = self.centers.unsqueeze(0).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        dist_between_centers = ((expanded_own_centers - expanded_centers)**2).sum(dim=2)  # (batch_size, num_classes)

        # 将正确类别的距离排除
        dist_between_other_centers = dist_between_centers * (1 - mask)
        sum_dist_between_centers = dist_between_other_centers.sum(dim=1)

        # 计算最终的对比损失
        loss = (dist_to_own_center / (sum_dist_between_centers + delta)).mean() * 0.5

        return loss

    def forward333(self, labels, x, delta=1e-6):
        #https://ar5iv.labs.arxiv.org/html/1707.07391
        batch_size = x.size(0)
        num_classes = self.centers.size(0)

        # 计算每个样本到其所属类别中心的距离 ||x_i - c_{y_i}||^2
        mask = F.one_hot(labels, num_classes).float()
        dist_to_own_center = ((x - self.centers[labels])**2).sum(dim=1)

        # 计算每个样本到所有其他类别中心的距离 ||x_i - c_j||^2
        expanded_x = x.unsqueeze(1).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        expanded_centers = self.centers.unsqueeze(0).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        dist_to_all_centers = ((expanded_x - expanded_centers)**2).sum(dim=2)  # (batch_size, num_classes)

        # 将正确类别的距离排除
        dist_to_other_centers = dist_to_all_centers * (1 - mask)
        sum_dist_to_other_centers = dist_to_other_centers.sum(dim=1)

        # 计算最终的对比损失
        loss = (dist_to_own_center / (sum_dist_to_other_centers + delta)).mean() * 0.5

        return loss

    def forward222(self, labels, x, delta=1e-6):
        '''
        使用余弦距离,代替欧氏距离
        '''

        batch_size = x.size(0)
        num_classes = self.centers.size(0)

        # 计算每个样本到其所属类别中心的距离 ||x_i - c_{y_i}||^2
        mask = F.one_hot(labels, num_classes).float()
        # dist_to_own_center = ((x -self.centers[labels]) ** 2).sum(dim=1)
        dist_to_own_center = torch.exp(F.cosine_similarity(x, self.centers[labels]))

        # 计算每个样本到所有其他类别中心的距离 ||x_i - c_j||^2
        expanded_x = x.unsqueeze(1).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        expanded_centers = self.centers.unsqueeze(0).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        # dist_to_all_centers = ((expanded_x - expanded_centers) ** 2).sum(dim=2)  # (batch_size, num_classes)
        dist_to_all_centers = torch.exp(F.cosine_similarity(expanded_x, expanded_centers, dim=2))

        # 将正确类别的距离排除
        dist_to_other_centers = dist_to_all_centers * (1 - mask)
        sum_dist_to_other_centers = dist_to_other_centers.sum(dim=1)

        # 计算最终的对比损失
        loss = -(dist_to_own_center / (sum_dist_to_other_centers + delta)).mean() * 0.5

        return loss



class CenterLoss2(nn.Module):
    """Center loss
    """

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        self.device = device


    def forward(self, labels, x, delta=1e-6):
        #https://ar5iv.labs.arxiv.org/html/1707.07391
        batch_size = x.size(0)
        num_classes = self.centers.size(0)

        # 计算每个样本到其所属类别中心的距离 ||x_i - c_{y_i}||^2
        mask = F.one_hot(labels, num_classes).float()
        dist_to_own_center = ((x - self.centers[labels])**2).sum(dim=1)

        # 计算每个样本到所有其他类别中心的距离 ||x_i - c_j||^2
        expanded_x = x.unsqueeze(1).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        expanded_centers = self.centers.unsqueeze(0).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        dist_to_all_centers = ((expanded_x - expanded_centers)**2).sum(dim=2)  # (batch_size, num_classes)

        # 将正确类别的距离排除
        dist_to_other_centers = dist_to_all_centers * (1 - mask)
        sum_dist_to_other_centers = dist_to_other_centers.sum(dim=1)

        # 计算最终的对比损失
        loss = (dist_to_own_center / (sum_dist_to_other_centers + delta)).mean() * 0.5

        return loss

    def forward222(self, labels, x, delta=1e-6):
        '''
        使用余弦距离,代替欧氏距离
        '''

        batch_size = x.size(0)
        num_classes = self.centers.size(0)

        # 计算每个样本到其所属类别中心的距离 ||x_i - c_{y_i}||^2
        mask = F.one_hot(labels, num_classes).float()
        # dist_to_own_center = ((x -self.centers[labels]) ** 2).sum(dim=1)
        dist_to_own_center = torch.exp(F.cosine_similarity(x, self.centers[labels]))

        # 计算每个样本到所有其他类别中心的距离 ||x_i - c_j||^2
        expanded_x = x.unsqueeze(1).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        expanded_centers = self.centers.unsqueeze(0).expand(batch_size, num_classes, -1)  # (batch_size, num_classes, feature_dim)
        # dist_to_all_centers = ((expanded_x - expanded_centers) ** 2).sum(dim=2)  # (batch_size, num_classes)
        dist_to_all_centers = torch.exp(F.cosine_similarity(expanded_x, expanded_centers, dim=2))

        # 将正确类别的距离排除
        dist_to_other_centers = dist_to_all_centers * (1 - mask)
        sum_dist_to_other_centers = dist_to_other_centers.sum(dim=1)

        # 计算最终的对比损失
        loss = -(dist_to_own_center / (sum_dist_to_other_centers + delta)).mean() * 0.5

        return loss


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
    mask_eq = torch.ones_like(similarity_matrix).to(device) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
    #这步得到它的不同类的矩阵，不同类的位置为1
    mask_not_eq = torch.ones_like(mask_eq).to(device) - mask_eq
    #这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_no_diag = torch.ones(n, n).to(device) - torch.eye(n, n).to(device)
    #这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix / T)
    #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix * mask_no_diag
    #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask_eq * similarity_matrix
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
    # loss = torch.sum(torch.sum(loss,dim=1)/torch.sum(loss!=0,dim=1))

    return loss


def supConLoss(features, labels, device="cuda", temperature=0.07, contrast_mode='all', base_temperature=0.07, mask=None):
    """
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:  # batch_size, channel,H,W，平铺变成batch_size, channel, (H,W)
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:  #只能存在一个
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:  #如果两个都没有就是无监督对比损失，mask就是一个单位阵
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:  #有标签，就把他变成mask
        labels = labels.contiguous().view(-1, 1)  #contiguous深拷贝，与原来的labels没有关系，展开成一列,这样的话能够计算mask，否则labels一维的话labels.T是他本身捕获发生转置
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  #label和label的转置比较，感觉应该是广播机制，让label和label.T都扩充了然后进行比较，相同的是1，不同是0.
        #这里就是由label形成mask,mask(i,j)代表第i个数据和第j个数据的关系，如果两个类别相同就是1， 不同就是0
    else:
        mask = mask.float().to(device)  #有mask就直接用mask，mask也是代表两个数据之间的关系

    contrast_count = features.shape[1]  #对比数是channel的个数
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  #把feature按照第1维拆开，然后在第0维上cat，(batch_size*channel,h*w..)#后面就是展开的feature的维度
    #这个操作就和后面mask.repeat对上了，这个操作是第一个数据的第一维特征+第二个数据的第一维特征+第三个数据的第一维特征这样排列的与mask对应
    if contrast_mode == 'one':  #如果mode=one，比较feature中第1维中的0号元素(batch, h*w)
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':  #all就(batch*channel, h*w)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),  #两个相乘获得相似度矩阵，乘积值越大代表越相关
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  #计算其中最大值
    logits = anchor_dot_contrast - logits_max.detach()  #减去最大值，都是负的了，指数就小于等于1

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  #repeat它就是把mask复制很多份
    # mask-out self-contrast cases
    logits_mask = torch.scatter(  #生成一个mask形状的矩阵除了对角线上的元素是0，其他位置都是1， 不会对自身进行比较
        torch.ones_like(mask), 1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  #定义其中的相似度
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  #softmax

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point.
    # Edge case e.g.:-
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan]
    mask_pos_pairs = mask.sum(1)  #mask的和
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  #满足返回1，不满足返回mask_pos_pairs.保证数值稳定
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos  #类似蒸馏temperature温度越高，分布曲线越平滑不易陷入局部最优解，温度低，分布陡峭
    loss = loss.view(anchor_count, batch_size).mean()  #计算平均

    return loss


if __name__ == '__main__':
    x = torch.randn((7, 1024)).to("cuda:0")
    y = torch.tensor([1, 2, 3, 1, 2, 3, 1]).to("cuda:0")
    # loss_val = supConLoss(x, y, "cuda:0")
    loss_val = supervisedContrastiveLoss(x, y, "cuda:0")
    print(loss_val)
