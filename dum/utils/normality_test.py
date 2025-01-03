#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   normality_test.py
@Time    :   2024/11/17 12:08:03
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import matplotlib.pyplot as plt
import numpy as np
from pingouin import multivariate_normality
from scipy.fft import fft, ifft
from scipy.stats import chi2, norm, normaltest, shapiro
from sklearn.decomposition import PCA


def energy_normality_test(data, alpha=0.05):
    # Step 1: Center the data
    n = len(data)
    mean_data = np.mean(data)
    centered_data = data - mean_data

    # Step 2: Compute the Fourier transform of the centered data
    f_data = fft(centered_data)

    # Step 3: Compute the energy statistic
    energy_stat = np.sum(np.abs(f_data)**2) / n  #越小越好

    # Step 4: Under the null hypothesis (data follows a normal distribution),
    # the expected energy statistic can be approximated as n.
    expected_energy = n

    # # Step 5: Compare the test statistic with the expected energy
    # # We can use a Chi-squared distribution approximation for the energy statistic under H0.
    p_value = 1 - norm.cdf(np.sqrt(energy_stat / expected_energy))

    return p_value, energy_stat


# 方法1：逐维正态性检测
def univariate_normality_test(data):
    """逐维检测正态性"""
    results = []
    stats = []
    for i in range(data.shape[1]):
        stat, p_value = normaltest(data[:, i])  # D’Agostino’s K^2 Test
        results.append(p_value)
        stats.append(stat)
    mean_p_value = np.mean(results)
    mean_stat = np.mean(stats)

    return mean_p_value, mean_stat


# 方法2：PCA降维后多元正态性检测
def pca_and_multivariate_normality_test(data, n_components=10):
    """PCA降维后检测多元正态性"""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    stat, p_value, normal = multivariate_normality(reduced_data, alpha=0.05)

    return p_value, stat


# 方法3：随机投影检测
def random_projection_normality_test(data, n_samples=100, n_features=10):
    """随机投影后检测正态性"""
    random_indices = np.random.choice(data.shape[1], n_features, replace=False)
    sampled_data = data[:, random_indices]

    return univariate_normality_test(sampled_data)


def normality_score(all_data, labels, method="univariate", num_classes=10):
    #p值越高，数据符合正态分布的可能性越高。正太性检验
    all_p_values = []
    all_stats = []  #统计量
    for i in range(num_classes):
        data = all_data[labels == i]

        if method == "pca":
            p_value, stats = pca_and_multivariate_normality_test(data)
        elif method == "univariate":
            p_value, stats = univariate_normality_test(data)
        elif method == "random":
            p_value, stats = random_projection_normality_test(data)
        elif method == "energy":
            p_value, stats = energy_normality_test(data)
        else:
            print("wrong method for normality detection")
            exit()

        all_p_values.append(p_value)
        all_stats.append(stats)

    return sum(all_p_values) / len(all_p_values), sum(all_stats) / len(all_stats)
