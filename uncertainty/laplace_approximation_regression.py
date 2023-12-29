#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   laplace_approximation.py
@Time    :   2023/11/07 15:59:29
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import sys
import time
from tqdm import tqdm

sys.path.append("../")

import numpy as np
import torch
import torch.distributions as dists
import torchvision.models as models
import torchvision.transforms as transforms
from laplace import Laplace
from laplace.curvature import AsdlGGN
from netcal.metrics import ECE
from torch import nn
from matplotlib import pyplot as plt

from model_utils.mlp import MLPNet
from utils.misc import argsdict, gen_data

def plot_regression(X_train,
                    y_train,
                    X_test,
                    f_test,
                    y_std,
                    plot=False,
                    file_name='regression_example'):
    fig, (ax1, ax2) = plt.subplots(nrows=1,
                                   ncols=2,
                                   sharey=True,
                                   figsize=(4.5, 2.8))
    ax1.set_title('MAP')
    ax1.scatter(X_train.flatten(),
                y_train.flatten(),
                alpha=0.3,
                color='tab:orange')
    ax1.plot(X_test, f_test, color='black', label='$f_{MAP}$')
    ax1.legend()

    ax2.set_title('LA')
    ax2.scatter(X_train.flatten(),
                y_train.flatten(),
                alpha=0.3,
                color='tab:orange')
    ax2.plot(X_test, f_test, label='$\mathbb{E}[f]$')
    ax2.fill_between(X_test,
                     f_test - y_std * 2,
                     f_test + y_std * 2,
                     alpha=0.3,
                     color='tab:blue',
                     label='$2\sqrt{\mathbb{V}\,[y]}$')
    ax2.legend()
    ax1.set_ylim([-4, 6])
    ax1.set_xlim([X_test.min(), X_test.max()])
    ax2.set_xlim([X_test.min(), X_test.max()])
    ax1.set_ylabel('$y$')
    ax1.set_xlabel('$x$')
    ax2.set_xlabel('$x$')
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(f'../results/laplace_regression/{file_name}.png')


def main():
    device = torch.device('cuda:1')
    model = MLPNet(0.5, True, 1, 32, 1)
    model = model.to(device)
    checkpoint = torch.load(
        "../saved_models/regression/deterministic/2023_12_26_14_29_19/mlp.pth",
        map_location=device)
    model.load_state_dict(checkpoint)

    MEAN_FUN = np.cos
    x_train, y_train, x_test, y_test = gen_data(
        mean_fun=MEAN_FUN,
        std_const=0.3,
        train_abs=6,
        test_abs=8,
        occlude=False,
        hetero=False,
        n_samples=2000,
    )
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=2000,
        shuffle=True,  # 将数据打乱
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=2000,
        shuffle=False,  # 将数据打乱
        num_workers=2,
    )

    la = Laplace(model,
                 'regression',
                 subset_of_weights='all',
                 hessian_structure='full')
    la.fit(train_loader)
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
        1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    n_epochs=1000
    for i in tqdm(range(n_epochs)):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(),
                                                  log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
    x = x_test.flatten().cpu().numpy()
    f_mu, f_var = la(x_test.to(device))
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

    plot_regression(x_train, y_train, x, f_mu, pred_std)


if __name__ == '__main__':
    main()
