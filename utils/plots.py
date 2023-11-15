#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plots.py
@Time    :   2023/11/13 16:59:42
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''



from sklearn import metrics
from matplotlib import pyplot as plt 


def plot_pr(y_true,y_score, label=None):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)  # the value of roc_auc1
    plt.plot(recall, precision, 'b', label=f'{label} AUPR = {aupr:.2f}')
    plt.legend(loc='upper right')
    plt.plot([1, 0], 'r--')
    plt.xlim([0, 1])  # the range of x-axis
    plt.ylim([0, 1])  # the range of y-axis
    plt.xlabel('Recall')  # the name of x-axis
    plt.ylabel('Precision')  # the name of y-axis
    plt.title('Precision-Recall')  # the title of figure
    plt.show()


def plot_roc(y_true, y_score,label=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auroc = metrics.auc(fpr, tpr)  # the value of roc_auc1
    plt.plot(fpr, tpr, 'b', label=f'{label} AUROC = {auroc:.2f}')
    plt.legend(loc='upper right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])  # the range of x-axis
    plt.ylim([0, 1])  # the range of y-axis
    plt.xlabel('False Positive Rate')  # the name of x-axis
    plt.ylabel('True Positive Rate')  # the name of y-axis
    plt.title('Receiver operating characteristic')  # the title of figure
    plt.show()
