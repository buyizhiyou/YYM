#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   plots.py
@Time    :   2023/11/13 16:59:42
@Author  :   shiqing
@Version :   Cinnamoroll V1
"""

import os
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn import metrics


def plot_pr(ax: Axes, y_true: list, y_score: list, label: str = None):
    """绘制PR曲线

    Args:
        ax (Axes): _description_
        y_true (list): _description_
        y_score (list): _description_
        label (str, optional): _description_. Defaults to None.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true, y_score)
    aupr = metrics.auc(recall, precision)  # the value of roc_auc1
    ax.plot(recall, precision, "b", label=f"{label} AUPR = {aupr:.2f}")
    ax.legend(loc="upper right")
    ax.plot([1, 0], "r--")
    ax.set_xlim([0, 1])  # the range of x-axis
    ax.set_ylim([0, 1])  # the range of y-axis
    ax.set_xlabel("Recall")  # the name of x-axis
    ax.set_ylabel("Precision")  # the name of y-axis
    ax.set_title("Precision-Recall")  # the title of figure


def plot_roc(ax: Axes, y_true: list, y_score: list, label: str = None):
    """绘制ROC曲线

    Args:
        ax (Axes): _description_
        y_true (list): _description_
        y_score (list): _description_
        label (str, optional): _description_. Defaults to None.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auroc = metrics.auc(fpr, tpr)  # the value of roc_auc1
    ax.plot(fpr, tpr, "b", label=f"{label} AUROC = {auroc:.2f}")
    ax.legend(loc="upper right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])  # the range of x-axis
    ax.set_ylim([0, 1])  # the range of y-axis
    ax.set_xlabel("False Positive Rate")  # the name of x-axis
    ax.set_ylabel("True Positive Rate")  # the name of y-axis
    ax.set_title("Receiver operating characteristic")  # the title of figure


def plot_dist(axes: Axes,
              data_list: List[list],
              datanames: List[str],
              label: str = None):
    """绘制分布

    Args:
        axes (Axes): _description_
        data_list (List[list]): _description_
        datanames (List[str]): _description_
        label (str, optional): _description_. Defaults to None.
    """
    sns.set_style("white")
    for i, data in enumerate(data_list):
        axes.set_xlabel(label)
        sns.kdeplot(
            data,
            ax=axes,
            label=datanames[i],
        )
    axes.legend()


def plot_ecdf(axes: Axes,
              data_list: List[list],
              datanames: List[str],
              label: str = None):
    """绘制cdf曲线

    Args:
        axes (Axes): _description_
        data_list (List[list]): _description_
        datanames (List[str]): _description_
        label (str, optional): _description_. Defaults to None.
    """
    sns.set_style("white")
    for i, data in enumerate(data_list):
        axes.set_xlabel(label)
        sns.ecdfplot(
            data,
            ax=axes,
            label=datanames[i],
        )
    axes.legend()


def plot_violin(axes: Axes,
                data_list: List[list],
                datanames: List[str],
                label: str = None):
    """绘制小提琴图

    Args:
        axes (Axes): _description_
        data_list (List[list]): _description_
        datanames (List[str]): _description_
        label (str, optional): _description_. Defaults to None.
    """
    sns.set_style("white")
    for i, data in enumerate(data_list):
        axes[i].set_xlabel(datanames[i])
        sns.violinplot(
            data,
            ax=axes[i],
        )
        axes[i].set_title(label)


def save_fig(fig: Figure, save_path: str, filename: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, filename))


def visualize_adv(
    x: np.ndarray,
    x_adv: np.ndarray,
    clean_pred: int,
    adv_pred: int,
    clean_prob: float,
    adv_prob: float,
    mean: list,
    std: list,
):
    """可视化对抗样本

    Args:
        x (np.ndarray): original image
        x_adv (np.ndarray): adversarial image
        clean_pred (int): original label
        adv_pred (int): new predictive label
        clean_prob (float): original prob
        adv_prob (float): new predictive prob
        mean (list): normalization mean
        std (list): normalization std
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    x = x * std + mean  # reverse of normalization op
    x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    x_adv = x_adv * std + mean  # reverse of normalization op
    x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    figure, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].imshow(x)
    ax[0].set_title("Clean Example", fontsize=10)

    ax[0].axis("off")
    ax[1].axis("off")

    ax[1].imshow(x_adv)
    ax[1].set_title("Adversarial Example", fontsize=10)

    ax[0].text(
        0.5,
        -0.2,
        "Pred: {}\n Prob: {:.2f}".format(clean_pred, clean_prob),
        size=10,
        ha="center",
        transform=ax[0].transAxes,
    )
    ax[1].text(
        0.5,
        -0.2,
        "Pred: {}\n Prob: {:.2f}".format(adv_pred, adv_prob),
        size=10,
        ha="center",
        transform=ax[1].transAxes,
    )

    plt.show()


def visualize_uncertainty(
    fig_title: str,
    pred_val: np.ndarray,
    ale_val: np.ndarray,
    epi_val: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 4))
    for ax in axs:
        ax.scatter(x_test, y_test, c="blue", s=7, label="test")
        ax.scatter(x_train, y_train, s=7, label="train", alpha=0.3)
        ax.plot(x_test, pred_val, c="red", label="predict")

    if np.mean(ale_val) != 1:  # a tricky hack here
        axs[0].fill_between(
            x_test.squeeze(),
            pred_val - ale_val,
            pred_val + ale_val,
            label="Aleatoric",
            alpha=0.5,
            color="orange",
        )
    else:
        axs[0].fill_between(  # TODO: S==0
            x_test.squeeze(),
            pred_val,
            pred_val,
            label="Aleatoric",
            alpha=0.5)

    if np.sum(epi_val) != 0:
        axs[1].fill_between(
            x_test.squeeze(),
            pred_val - epi_val,
            pred_val + epi_val,
            label="Epistemic",
            alpha=0.5,
            color="orange",
        )
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(fig_title)
    plt.show()

    return fig
