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


def evaluate_roc_pr(y_true, y_score):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auroc = metrics.auc(fpr, tpr)  # the value of roc_auc1
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true, y_score)
    aupr = metrics.auc(recall, precision)  # the value of roc_auc1
    return auroc, aupr


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
                  device="cuda:0"):
    faiss_gpu_id = -1
    feature_dim = train_embeddings.shape[1]
    sigma = 0.1


    if (faiss_gpu_id >= 0):
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatL2(feature_dim)  # 这里传入向量的维度，创建一个空的索引
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_id, index_flat)
    else:
        index = faiss.IndexFlatL2(feature_dim)

    index.add(train_embeddings.cpu().numpy().astype("float32"))  # 把向量数据加入索引
    import pdb;pdb.set_trace()
    train_log_prob = torch.logsumexp(prob_model.log_prob(train_embeddings[:, None, :]),dim=1)

    dist, idxs = index.search(test_embeddings.cpu().numpy().astype("float32"),k=100)
    dist = dist / np.min(dist, axis=1)[:, None]
    ood_dist, ood_idxs = index.search(ood_test_embeddings.cpu().numpy().astype("float32"),k=100)  #返回numpy array
    ood_dist = ood_dist / np.min(ood_dist, axis=1)[:, None]


    dist = torch.from_numpy(dist)
    w = torch.exp(dist - (-1)**2 / (2 * sigma**2)).to(device)
    log_probs = train_log_prob[idxs]
    wde = (w * log_probs).sum(dim=1) / torch.sum(w)
    kde = torch.logsumexp(prob_model.log_prob(test_embeddings[:, None, :]),dim=1)
    ind_af = wde / kde

    ood_dist = torch.from_numpy(ood_dist)
    ood_w = torch.exp(-(ood_dist - 1)**2 / (2 * sigma**2)).to(device)
    ood_log_probs = train_log_prob[ood_idxs]
    ood_wde = (ood_w * ood_log_probs).sum(dim=1) / torch.sum(ood_w)
    ood_kde = torch.logsumexp(prob_model.log_prob(ood_test_embeddings[:, None, :]),dim=1)
    ood_af = ood_wde / ood_kde

    labels = [1] * (test_embeddings.shape[0]) + [0] * (ood_test_embeddings.shape[0])
    scores = list(ind_af.cpu().numpy()) + list(ood_af.cpu().numpy())
    auroc, aupr = evaluate_roc_pr(labels, scores)
    print(f"sigma:{sigma},auroc:{auroc},aupr:{aupr}")

    import pdb
    pdb.set_trace()


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
