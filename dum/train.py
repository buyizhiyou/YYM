"""
Script for training a single model for OOD detection.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tqdm import tqdm
import warmup_scheduler
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import data_utils.dirty_mnist as dirty_mnist
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.mnist as mnist
from net.lenet import lenet
from net.resnet import resnet18, resnet50  # 自己实现的spectral norm
# from net.resnet2 import resnet18, resnet50 #官方实现的spectral norm
from net.vgg import vgg16  # 自己实现的
from net.vit import vit
# from net.vgg2 import vgg16 #官方实现的
from net.wide_resnet import wrn
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.lars import LARC
from utils.loss import CenterLoss, LabelSmoothing, supervisedContrastiveLoss, GMMRegularizationLoss
from utils.normality_test import normality_score
from utils.plots_utils import (create_gif_from_images, inter_intra_class_ratio, plot_embedding_2d)
from utils.train_utils import (model_save_name, save_config_file, test_single_epoch, train_single_epoch, seed_torch)

dataset_num_classes = {"mnist": 10, "cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn, "dirty_mnist": dirty_mnist, "mnist": mnist}

models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}
model_to_num_dim = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048, "wide_resnet": 640, "vgg16": 512, "vit": 768}

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    args = training_args().parse_args()
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print("CUDA set: " + str(cuda))
    size = args.size
    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes).to(device)
    net.to(device)

    opt_params = net.parameters()
    # 设置optimier
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 学习率schduler
    if args.scheduler == "step":
        ##150,250
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.first_milestone, args.second_milestone],  #150,250
            #    milestones=[0.3 * args.epoch, 0.6 * args.epoch, 0.9 * args.epoch],
            gamma=0.1,
            verbose=False)
    elif args.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-7, verbose=False)

    # scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)
    #TODO:这个schduler有Bug，无法step更新学习率
    # optimimizer = LARC(optimizer)

    if args.contrastive == 3:
        aux_loss = CenterLoss(num_classes=10, feat_dim=model_to_num_dim[args.model], device=device)
    elif args.contrastive == 4:
        aux_loss = GMMRegularizationLoss(num_classes=10, feature_dim=model_to_num_dim[args.model], device=device)
    else:
        aux_loss = None

    if args.contrastive == 3 or args.contrastive == 4:
        optimizer_auxloss = torch.optim.SGD(aux_loss.parameters(), lr=0.5)
        # scheduler_auxloss = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1, verbose=False)
    else:
        optimizer_auxloss = None

    #设置dataloader
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(root=args.dataset_root,
                                                                                   batch_size=args.train_batch_size,
                                                                                   augment=args.data_aug,
                                                                                   val_size=0.1,
                                                                                   val_seed=args.seed,
                                                                                   pin_memory=args.gpu,
                                                                                   contrastive=args.contrastive,
                                                                                   size=size)
    test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=args.train_batch_size, size=size)
    test_loader2 = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=32, sample_size=100000, size=size)
    ood_test_loader = svhn.get_test_loader(32, root="./data/", sample_size=2000, size=size)

    # Creating summary writer in tensorboard
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
    save_name = model_save_name(args.model, args.sn, args.mod, args.seed, args.contrastive)
    print("Model save name", save_name)

    if args.ls:
        save_loc = f"{args.save_loc}/run{args.run}/{save_name}/{time_str}_labelsmooth/"
    else:
        save_loc = f"{args.save_loc}/run{args.run}/{save_name}/{time_str}/"  # 保存模型路径
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    log_loc = f"{args.log_loc}/run{args.run}/{save_name}/{time_str}"
    writer = SummaryWriter(log_loc)
    save_config_file(save_loc, args)

    best_acc = 0
    for epoch in tqdm(range(0, args.epoch)):
        """
        1. 300epoch 原始单阶段训练crossEntropy
        2. 两阶段训练:前300epoch只训练supCon,后面150个epoch只训练fc层
        """
        print("best accu", best_acc)
        train_loss, train_acc = train_single_epoch(
            epoch,
            net,
            train_loader,
            optimizer,
            aux_loss,
            optimizer_auxloss,
            device,
            args.contrastive,
            adv=args.adv,
            label_smooth=args.ls,
        )
        scheduler.step()
        # scheduler_auxloss.step()

        if epoch % 3 == 0:
            val_acc = test_single_epoch(epoch, net, val_loader, device)
            writer.add_scalar("train_loss", train_loss, (epoch + 1))
            writer.add_scalar("train_acc", train_acc, (epoch + 1))
            writer.add_scalar("val_acc", val_acc, (epoch + 1))
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=(epoch + 1))

        if epoch < (args.epoch - 5):
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = save_loc + save_name + "_best" + ".model"
                torch.save(net.state_dict(), save_path)
                Path(os.path.join(save_loc, f"accuracy_{best_acc}")).touch()
                print("Model saved to ", save_path)
                # if args.contrastive == 4 or args.contrastive == 3:  #对于centerloss 或者gmmloss还需要额外保存loss的参数
                #     save_path2 = save_loc + save_name + "_best" + "_gmm.model"
                #     torch.save(aux_loss.state_dict(), save_path2)
        else:  #在最后10个epoch,acc已经基本平直
            Xs = []
            ys = []
            for images, labels in test_loader2:
                images = images.to(device)
                _ = net(images)
                embeddings = net.feature
                Xs.append(embeddings.cpu().detach().numpy())
                ys.append(labels.detach().numpy())
            X = np.concatenate(Xs)
            y = np.concatenate(ys)

            for images, _ in ood_test_loader:
                labels = np.ones(images.shape[0]) * 10  #标记label=10为OOD样本
                images = images.to(device)
                _ = net(images)
                embeddings = net.feature
                Xs.append(embeddings.cpu().detach().numpy())
                ys.append(labels)

            X = np.concatenate(Xs)
            y = np.concatenate(ys)
            tsne = TSNE(n_components=2, init='pca', perplexity=50, random_state=0)
            X_tsne = tsne.fit_transform(X)

            fig = plot_embedding_2d(X_tsne, y, 10, f"epoch:{epoch},stats:{0.0:.3f}")
            fig.savefig(os.path.join(save_loc, f"stats_{epoch}.jpg"), dpi=50, bbox_inches='tight')

            # save_path = save_loc + save_name + f"_epoch_{epoch}" + ".model"
            # torch.save(net.state_dict(), save_path)

    writer.close()
    # create_gif_from_images(save_loc)
