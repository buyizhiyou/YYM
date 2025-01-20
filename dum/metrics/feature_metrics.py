#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   feature_metrics.py
@Time    :   2025/01/10 19:47:22
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score

# ========================
# 类内紧密性和类间可分性度量工具
# ========================


# 1. 类内距离 (Intra-class Distance)
def intra_class_distance(features, labels):
    """
    计算类内距离的均值。较小的类内距离表示类内紧密性更好
    :param features: 样本特征 (N, D)，N为样本数，D为特征维度
    :param labels: 样本标签 (N,)
    :return: 类内平均距离
    """
    classes = np.unique(labels)  # 获取所有类别
    distances = []

    for c in classes:
        class_features = features[labels == c]  # 取出属于类别 c 的样本
        center = np.mean(class_features, axis=0)  # 计算类别中心
        distances.extend(np.linalg.norm(class_features - center, axis=1))  # 距离

    return round(np.mean(distances), 4)  # 平均类内距离


# 2. 类内方差 (Intra-class Variance)
def intra_class_variance(features, labels):
    """
    计算类内样本特征的方差。较小的类内方差表示类内紧密性更好
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: 类内方差
    """
    classes = np.unique(labels)
    total_variance = 0

    for c in classes:
        class_features = features[labels == c]
        variance = np.var(class_features, axis=0)  # 每个特征的方差
        total_variance += np.sum(variance)  # 累积方差

    return round(total_variance / len(classes), 4)  # 平均方差


# 3. Davies-Bouldin Index (DBI)
def davies_bouldin_index(features, labels):
    """
    计算 Davies-Bouldin 指数。
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: DBI 值，越小越好，综合考虑了类内紧密性和类间可分性
    """
    classes = np.unique(labels)
    centers = []
    intra_dists = []

    # 计算每个类的中心和类内距离
    for c in classes:
        class_features = features[labels == c]
        center = np.mean(class_features, axis=0)
        centers.append(center)
        intra_dists.append(np.mean(np.linalg.norm(class_features - center, axis=1)))  #每个类的类内距离

    centers = np.array(centers)
    dbi = 0

    for i in range(len(classes)):
        max_ratio = 0
        for j in range(len(classes)):
            if i != j:
                inter_dist = np.linalg.norm(centers[i] - centers[j])  # 类间距离，簇i和j的中心之间的欧几里得距离（类间分离度）。
                ratio = (intra_dists[i] + intra_dists[j]) / inter_dist  # 类内-类间比率
                max_ratio = max(max_ratio, ratio)
        dbi += max_ratio

    return round(dbi / len(classes), 4)


# 4. 类间距离 (Inter-class Distance)
def inter_class_distance(features, labels):
    """
    计算类别中心之间的距离，较大的距离表示类间更分离
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: 类间平均距离
    """
    classes = np.unique(labels)
    centers = []

    for c in classes:
        class_features = features[labels == c]
        center = np.mean(class_features, axis=0)
        centers.append(center)

    centers = np.array(centers)
    distances = []

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distances.append(np.linalg.norm(centers[i] - centers[j]))  # 中心之间的距离

    return round(np.mean(distances), 4)


# 5. Fisher 比率 (Fisher Ratio)
def fisher_ratio(features, labels):
    """
    计算 Fisher 比率。
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: Fisher 比率 = 类间散布/类内散布
    较大的 Fisher 比率表明类间分离性更强，类内紧密性更高。
    """
    classes = np.unique(labels)
    overall_mean = np.mean(features, axis=0)  # 整体均值
    sw = 0  # 类内散布矩阵
    sb = 0  # 类间散布矩阵

    for c in classes:
        class_features = features[labels == c]
        class_mean = np.mean(class_features, axis=0)
        n_c = class_features.shape[0]

        # 类内散布
        sw += np.sum((class_features - class_mean)**2)

        # 类间散布
        sb += n_c * np.sum((class_mean - overall_mean)**2)

    return round(sb / sw, 4)


# 6. Silhouette Coefficient (轮廓系数)
def silhouette_coefficient(features, labels):
    """
    计算 Silhouette 系数。
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: Silhouette 系数，范围 [-1, 1]
    值接近 1：表示样本很好地划分到了正确的簇，簇内紧密且簇间分离。
    值接近 0：表示样本位于簇的边界，可能难以归类。
    值接近 −1：表示样本更接近其他簇，聚类划分可能存在问题。
    
    """
    n_samples = len(features)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # 验证样本条件
    if n_clusters <= 1 or n_clusters >= n_samples:
        raise ValueError("Silhouette Coefficient 需要至少有两个簇且每簇至少有一个样本。")

    # 计算距离矩阵
    distances = pairwise_distances(features)  # 使用欧几里得距离

    # 初始化 a 和 b
    a = np.zeros(n_samples)  # 类内平均距离
    b = np.full(n_samples, np.inf)  # 类间最近距离

    # 每个簇的样本索引和数量
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # 计算 a(i) 和 b(i)
    # a(i):当前样本 i 到同一簇内其他样本的平均距离，衡量簇内的凝聚力。
    # b(i): 类间可分性是样本 i 与最接近的其他簇的所有样本的平均距离。它衡量了样本与其他簇的分离程度。
    for label, indices in cluster_indices.items():
        # 类内平均距离 a(i)
        intra_distances = distances[np.ix_(indices, indices)]  # 当前簇的距离矩阵
        a[indices] = np.sum(intra_distances, axis=1) / (len(indices) - 1)  # 类内距离

        # 计算每个样本到其他簇的距离
        for other_label, other_indices in cluster_indices.items():
            if label == other_label:
                continue  # 跳过同簇
            inter_distances = np.mean(distances[np.ix_(indices, other_indices)], axis=1)
            b[indices] = np.minimum(b[indices], inter_distances)  # 最近簇的距离

    # 计算轮廓系数 s(i)
    s = (b - a) / np.maximum(a, b)
    s[a == 0] = 0  # 避免 a=0 时的 NaN

    # 返回平均 Silhouette 系数
    return round(np.mean(s), 4)


# 7. 整合为工具函数
def evaluate_class_separability(features, labels):
    """
    综合评估类内紧密性和类间可分性。
    :param features: 样本特征 (N, D)
    :param labels: 样本标签 (N,)
    :return: 指标字典
    """
    metrics = {
        "Intra-class Distance": intra_class_distance(features, labels),
        "Intra-class Variance": intra_class_variance(features, labels),
        "Davies-Bouldin Index": davies_bouldin_index(features, labels),
        "Inter-class Distance": inter_class_distance(features, labels),
        "Fisher Ratio": fisher_ratio(features, labels),
        "Silhouette Coefficient": silhouette_score(features, labels),
    }
    return metrics


# # 示例测试
# if __name__ == "__main__":
#     # 示例数据
#     features = np.array([[1.0, 2.0], [1.1, 2.1], [3.0, 3.0], [3.1, 3.1]])
#     labels = np.array([0, 0, 1, 1])

#     # 评估指标
#     results = evaluate_class_separability(features, labels)
#     for name, value in results.items():
#         print(f"{name}: {value:.4f}")
