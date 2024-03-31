#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lof.py
@Time    :   2024/03/27 16:36:34
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''
"实现基于密度的异常因子的几种评估方式"

import torch
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn import metrics
import numpy as np
import faiss


def evaluate_roc_pr(labels, scores):
    auroc = metrics.roc_auc_score(labels, scores)
    auprc = metrics.average_precision_score(labels, scores)

    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    # auroc = metrics.auc(fpr, tpr)  # the value of roc_auc1
    # precision, recall, thresholds = metrics.precision_recall_curve(
    #     labels, scores)
    # aupr = metrics.auc(recall, precision)  # the value of roc_auc1
    return auroc, auprc


def lof_evaluate(train_embeddings, test_embeddings, ood_test_embeddings):
    for neighbors in range(10, 100, 10):
        clf = LOF(n_neighbors=neighbors, novelty=True)
        clf.fit(train_embeddings)
        ind_scores = clf.score_samples(test_embeddings)
        ood_scores = clf.score_samples(ood_test_embeddings)

        labels = [1] * (ind_scores.shape[0]) + [0] * (ood_scores.shape[0])
        scores = list(ind_scores) + list(ood_scores)
        auroc, aupr = evaluate_roc_pr(labels, scores)
        print(f"neighbors:{neighbors},auroc:{auroc},aupr:{aupr}")

    return scores


def ldaf_evaluate(prob_model,
                  train_embeddings,
                  test_embeddings,
                  ood_test_embeddings,
                  device="cuda:0",
                  faiss_gpu_id=1,
                  sigma=0.01,
                  k=100,
                  conf=True):
    """local density-based anomaly factor

    Args:
        prob_model (_type_): 建好的概率模型
        train_embeddings (_type_): _description_
        test_embeddings (_type_): _description_
        ood_test_embeddings (_type_): _description_
        device (str, optional): _description_. Defaults to "cuda:0".
        faiss_gpu_id (int, optional): faiss使用的显卡id，如果小于0，将使用cpu. Defaults to 1.
        sigma (float, optional): _description_. Defaults to 0.1.
        k (int, optional): 搜索的k个最近的邻居. Defaults to 100.

    Returns:
        _type_: _description_
    """
    feature_dim = train_embeddings.shape[1]

    if (faiss_gpu_id >= 0):
        res = faiss.StandardGpuResources()  # use a single GPU for Faiss
        index_flat = faiss.IndexFlatL2(feature_dim)  # 传入向量的维度，创建一个空的索引
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_id, index_flat)
    else:
        index = faiss.IndexFlatL2(feature_dim)

    index.add(train_embeddings.cpu().numpy().astype("float32"))  # 把向量数据加入索引
    train_log_prob_list = []

    for i in range(5):  #分批次计算，否则爆显存
        train_log_prob_list.append(torch.logsumexp(prob_model.log_prob(train_embeddings[i * 10000:(i + 1) * 10000, None, :]),dim=1))
    train_log_prob = torch.concat(train_log_prob_list)


    dist, idxs = index.search(test_embeddings.cpu().numpy().astype("float32"),
                            k=k)
    dist = dist / np.min(dist, axis=1)[:, None]
    ood_dist, ood_idxs = index.search(
        ood_test_embeddings.cpu().numpy().astype("float32"),
        k=k)  #返回numpy array
    ood_dist = ood_dist / np.min(ood_dist, axis=1)[:, None]

    dist = torch.from_numpy(dist)
    w = torch.exp(dist - (-1)**2 / (2 * sigma**2)).to(device) #计算权重
    log_probs = train_log_prob[idxs]
    wde = (w * log_probs).sum(dim=1) / torch.sum(w)
    kde = torch.logsumexp(prob_model.log_prob(test_embeddings[:, None, :]),
                        dim=1)
    ind_af = wde / kde  #指标越高，

    ood_dist = torch.from_numpy(ood_dist)
    ood_w = torch.exp(-(ood_dist - 1)**2 / (2 * sigma**2)).to(device)
    ood_log_probs = train_log_prob[ood_idxs]
    ood_wde = (ood_w * ood_log_probs).sum(dim=1) / torch.sum(ood_w)
    ood_kde = torch.logsumexp(prob_model.log_prob(
        ood_test_embeddings[:, None, :]),
                            dim=1)
    ood_af = ood_wde / ood_kde  #理论上来说,ood的这个指标应该更高一点

    labels = np.concatenate([np.ones(test_embeddings.shape[0]),np.zeros(ood_test_embeddings.shape[0])])
    scores = np.concatenate([(ind_af.cpu().numpy()) , list(ood_af.cpu().numpy())])

    scores =  np.nan_to_num(scores)

    if conf: #True时，指标越高，信心越低，不确定性越高，这时将ood样本置为1
        labels = 1 - labels
    auroc, aupr = evaluate_roc_pr(labels, scores)
    
    print(f"k_neighbors:{k},sigma:{sigma},auroc:{auroc},aupr:{aupr}")
    
    return auroc,aupr

if __name__ == '__main__':
    device = "cuda:0"
    covariance_matrix = (torch.zeros(
        (10, 2048, 2048)) + torch.eye(2048)).to(device)
    loc = torch.randn((10, 2048)).to(device)
    gmm = torch.distributions.MultivariateNormal(loc, covariance_matrix)

    x = torch.randn((50000, 2048)).to(device)
    y = torch.randn((10000, 2048)).to(device)
    z = torch.randn((20000, 2048)).to(device)
    ldaf_evaluate(gmm, x, y, z)
