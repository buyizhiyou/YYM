#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_regression_bayesian.py
@Time    :   2023/12/20 22:00:25
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import argparse
import datetime
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from model_utils.mlp import MLPNet
from utils.loss import AleatoricLoss
from utils.misc import argsdict, gen_data
from utils.visual import AverageMeter, ProgressMeter, Summary
from utils.early_stopping import EarlyStopping

parser = argparse.ArgumentParser(description='Training')
parser.add_argument(
    '--config', default="./config/regression/mlp_bayesian.yaml", help='yaml config file')


def main():
    configargs = parser.parse_args()
    with open(configargs.config, "r") as f:
        cfg = yaml.safe_load(f)
    args = argsdict(cfg)  # 包装字典，可以通过.访问

    # def MEAN_FUN(x): return x**3
    MEAN_FUN = np.cos
    x_train, y_train, x_test, y_test = gen_data(
        mean_fun=MEAN_FUN, std_const=args.std_const, train_abs=args.train_abs, test_abs=args.test_abs,
        occlude=args.occlude, hetero=args.hetero,n_samples=args.n_samples)

    # train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    # val_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,  # 将数据打乱
    #     num_workers=2,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,  # 将数据打乱
    #     num_workers=2,
    # )

    device = f"cuda:{args.gpu}"
    aleatoric_loss = AleatoricLoss()

    net = MLPNet(p=args.drop_p, logvar=True, n_feature=1,
                 n_hidden=args.n_hidden, n_output=1).to(device)
    moped_enable = False
    if len(args.moped_init_model) > 0:  # use moped method if trained dnn model weights are provided
        moped_enable = True
    const_bnn_prior_parameters = {
        "prior_mu": args.prior_mu,
        "prior_sigma": args.prior_sigma,
        "posterior_mu_init": args.posterior_mu_init,
        "posterior_rho_init": args.bnn_rho_init,
        # Flipout or Reparameterization
        "type": "Flipout" if args.use_flipout_layers else "Reparameterization",
        "moped_enable": moped_enable,  # initialize mu/sigma from the dnn weights
        "moped_delta": args.moped_delta_factor,
    }
    dnn_to_bnn(net, const_bnn_prior_parameters)
    net.to(device)

    optimizer = optim.Adam(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")

    log_dir = f"{args.logs}/{args.type}/{args.mode}/{time_str}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    model_dir = f"{args.saved_models}/{args.type}/{args.mode}/{time_str}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.occlude and args.hetero:
        save_path = os.path.join(model_dir, f"mlp_occlude_hetero.pth")
    elif args.hetero:
        save_path = os.path.join(model_dir, f"mlp_hetero.pth")
    elif args.occlude:
        save_path = os.path.join(model_dir, f"mlp_occlude.pth")
    else:
        save_path = os.path.join(model_dir, f"mlp.pth")
    early_stopping = EarlyStopping(save_path, patience=10000)
    net = train_model(net, aleatoric_loss, optimizer, x_train,
                      y_train, x_test, y_test, device, early_stopping, writer, args.number_epochs)

    with open("logs/model_parameters_map.yaml", "a") as f:  # 保存模型日期和训练参数
        yaml.dump({f"mlp_{args.mode}_{time_str}": dict(args)}, f)
    
    return net


def train_model(
    network,
    loss_fun,
    optimizer,
    x_train,
    y_train,
    x_test,
    y_test,
    device,
    early_stopping,
    writer,
    number_epochs=1000,
    num_mc_train=1,
    num_mc_eval=20
):
    for epoch in tqdm(range(number_epochs)):
        # losses = AverageMeter('Loss', ':.4e')
        # for x_train, y in train_loader:
        network.train()
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        pred_ = []
        logvar_ = []
        kl_ = []
        for mc_run in range(num_mc_train):
            pred, logvar = network(x_train)
            kl = get_kl_loss(network)
            pred_.append(pred)
            logvar_.append(logvar)
            kl_.append(kl)
        pred = torch.mean(torch.stack(pred_), dim=0)
        logvar = torch.mean(torch.stack(logvar_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        loss = loss_fun(y_train, pred, logvar.to(device))

        alpha = 10
        total_loss = loss+ alpha*kl

        optimizer.zero_grad()  # clear gradients for next train
        total_loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        network.eval()
        with torch.no_grad():
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pred_ = []
            logvar_ = []
            for mc_run in range(num_mc_eval):
                pred, logvar = network(x_test)
                pred_.append(pred)
                logvar_.append(logvar)
            pred = torch.mean(torch.stack(pred_), dim=0)
            logvar = torch.mean(torch.stack(logvar_), dim=0)
            val_loss = loss_fun(y_train, pred, logvar.to(device))

        if epoch % 10000 == 0:
            print(
                f'Epoch {epoch+1}/{number_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        # losses.update(loss.item(), x.size(0))
        writer.add_scalar("Loss/train", loss,  epoch)
        writer.add_scalar("Loss/val", val_loss,  epoch)
        writer.add_scalar("Loss/total", total_loss,  epoch)
        writer.add_scalar("Loss/kl", kl,  epoch)

        early_stopping(val_loss, network)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    return network


if __name__ == '__main__':
    main()
