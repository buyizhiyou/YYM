#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lof.py
@Time    :   2024/03/27 16:36:34
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys
sys.path.append("../")

from metrics.ood_metrics import fpr_at_95_tpr
import faiss
import numpy as np
import torch
from utils.gmm_utils import gmm_fit
from utils.kde_utils import kde_fit, KdeModel
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor as LOF
from tqdm import tqdm
import time
"实现基于密度的异常因子的几种评估方式"


def evaluate_roc_pr(labels, scores):
    auroc = metrics.roc_auc_score(labels, scores)
    auprc = metrics.average_precision_score(labels, scores)

    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    # auroc = metrics.auc(fpr, tpr)  # the value of roc_auc1
    # precision, recall, thresholds = metrics.precision_recall_curve(
    #     labels, scores)
    # aupr = metrics.auc(recall, precision)  # the value of roc_auc1
    return auroc, auprc


def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)


def sumexp(logits):
    return torch.sum(torch.exp(logits), dim=1)


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
                  sigma=0.1,
                  k=100,
                  cos=False,
                  conf=True):
    """local density based anomaly factor

    Args:
        prob_model (_type_): _description_
        train_embeddings (_type_): _description_
        test_embeddings (_type_): _description_
        ood_test_embeddings (_type_): _description_
        device (str, optional): _description_. Defaults to "cuda:0".
        faiss_gpu_id (int, optional): _description_. Defaults to 1.
        sigma (float, optional): _description_. Defaults to 0.1.
        k (int, optional): _description_. Defaults to 100.
        cos (bool, optional): _description_. Defaults to False.
        conf (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    for k in [100]:
        for sigma in [0.1]:
            feature_dim = train_embeddings.shape[1]

            if isinstance(prob_model, KdeModel):
                start = time.time()
                train_log_prob = logsumexp(torch.from_numpy(prob_model.log_prob(train_embeddings.cpu())).to(device))
                end = time.time()
                print(f"kde train execution time:{end-start:.1f}")
            else:
                train_log_prob_list = []
                for i in range(5):  # 分批次计算，否则爆显存
                    train_log_prob_list.append(logsumexp(prob_model.log_prob(train_embeddings[i * 10000:(i + 1) * 10000, None, :])))
                train_log_prob = torch.concat(train_log_prob_list)

            if isinstance(prob_model, KdeModel):
                start = time.time()
                pdf = logsumexp(torch.from_numpy(prob_model.log_prob(test_embeddings.cpu())).to(device))
                end = time.time()
                print(f"kde test execution time:{end-start:.1f}")
            else:
                pdf = logsumexp(prob_model.log_prob(test_embeddings[:, None, :]))

            if isinstance(prob_model, KdeModel):
                start = time.time()
                ood_pdf = logsumexp(torch.from_numpy(prob_model.log_prob(ood_test_embeddings.cpu())).to(device))
                end = time.time()
                print(f"kde  ood test execution time:{end-start:.1f}")
            else:
                ood_pdf = logsumexp(prob_model.log_prob(ood_test_embeddings[:, None, :]))

            train_embeddings = train_embeddings.cpu().numpy().astype("float32")
            test_embeddings = test_embeddings.cpu().numpy().astype("float32")
            ood_test_embeddings = ood_test_embeddings.cpu().numpy().astype("float32")
            if cos:  #L2归一化
                train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
                test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
                ood_test_embeddings = ood_test_embeddings / np.linalg.norm(ood_test_embeddings, axis=1, keepdims=True)

            if (faiss_gpu_id >= 0):
                gpu = faiss.StandardGpuResources()  # use a single GPU for Faiss
                if cos:
                    index_flat = faiss.IndexFlatIP(feature_dim)  #内积相似度，为了计算cos相似度，首先需要归一化特征
                else:
                    index_flat = faiss.IndexFlatL2(feature_dim)  # 传入向量的维度，创建一个空的索引
                index = faiss.index_cpu_to_gpu(gpu, faiss_gpu_id, index_flat)
            else:
                index = faiss.IndexFlatL2(feature_dim)

            index.add(train_embeddings)  # 把向量数据加入索引

            _, idxs = index.search(test_embeddings, k=k)  # N_k(p)
            neighbors = train_embeddings[idxs].reshape(-1, feature_dim)
            neighbors_dist, _ = index.search(neighbors, k=k)
            dist = np.max(neighbors_dist, axis=1).reshape((-1, k))
            dist = dist / np.min(dist, axis=1)[:, None]

            _, ood_idxs = index.search(ood_test_embeddings, k=k)  # 返回numpy array
            ood_neighbors = train_embeddings[ood_idxs].reshape(-1, feature_dim)
            ood_neighbors_dist, _ = index.search(ood_neighbors, k=k)
            ood_dist = np.max(ood_neighbors_dist, axis=1).reshape((-1, k))
            ood_dist = ood_dist / np.min(ood_dist, axis=1)[:, None]

            dist = torch.from_numpy(dist)
            w = torch.exp(-(dist - 1)**2 / (2 * sigma**2)).to(device)  # 计算权重
            log_probs = train_log_prob[torch.from_numpy(idxs)]
            wpdf = (w * log_probs).sum(dim=1) / torch.sum(w)

            ind_af = wpdf / pdf  # 指标越高，

            ood_dist = torch.from_numpy(ood_dist)
            ood_w = torch.exp(-(ood_dist - 1)**2 / (2 * sigma**2)).to(device)
            ood_log_probs = train_log_prob[torch.from_numpy(ood_idxs)]
            ood_wpdf = (ood_w * ood_log_probs).sum(dim=1) / torch.sum(ood_w)

            ood_af = ood_wpdf / ood_pdf  # 理论上来说,ood的这个指标应该更高一点

            labels = np.concatenate([np.ones(test_embeddings.shape[0]), np.zeros(ood_test_embeddings.shape[0])])
            scores = np.concatenate([ind_af.cpu().numpy(), ood_af.cpu().numpy()])
            # scores = np.nan_to_num(scores)
            scores2 = np.concatenate([pdf.cpu().numpy(), ood_pdf.cpu().numpy()])

            if conf:  # True时，指标越高，信心越低，不确定性越高，这时将ood样本置为1
                labels = 1 - labels
            auroc = metrics.roc_auc_score(labels, scores)
            auprc = metrics.average_precision_score(labels, scores)
            fpr95 = fpr_at_95_tpr(scores,labels)

            auroc2 = metrics.roc_auc_score(1 - labels, scores2)
            auprc2 = metrics.average_precision_score(1 - labels, scores2)
            fpr95_2 = fpr_at_95_tpr(scores, 1 - labels)

            print(f"k_neighbors:{k},sigma:{sigma},auroc:{auroc:.4f},auprc:{auprc:.4f},fpr95:{fpr95}")
            print(f"k_neighbors:{k},sigma:{sigma},auroc2:{auroc2:.4f},auprc2:{auprc2:.4f},fpr95:{fpr95_2}")

    return auroc, auprc


# test
# if __name__ == '__main__':
#     device = "cuda:0"
#     train_embeddings = torch.load("train_embeddings.pth")
#     test_embeddings = torch.load("test_embeddings.pth")
#     ood_test_embeddings = torch.load("ood_test_embeddings.pth")
#     labels = torch.load("labels.pth")
#     gaussians_model, jitter_eps = gmm_fit(embeddings=train_embeddings, labels=labels, num_classes=10)
#     # kde_model = kde_fit(train_embeddings, labels, 10)
#     #k_neighbors:100,sigma:0.1,auroc:0.1913,auprc:0.5535
#     # k_neighbors:100,sigma:0.1,auroc2:0.8755,auprc2:0.8710

#     ldaf_evaluate(gaussians_model,
#                   train_embeddings,
#                   test_embeddings,
#                   ood_test_embeddings,
#                   device="cuda:0",
#                   faiss_gpu_id=1,
#                   sigma=0.1,
#                   k=100,
#                   cos=False,
#                   conf=True)
