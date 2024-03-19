"""
Script for training a single model for OOD detection.
"""

import datetime
import json
import os
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
import data_utils.dirty_mnist as dirty_mnist

# Import network models
from net.lenet import lenet
# from net.resnet import resnet18, resnet50, resnet101, resnet110, resnet152
from net.resnet2 import resnet18, resnet50, resnet101, resnet110, resnet152
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "dirty_mnist": 10
}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet110": resnet110,
    "resnet152": resnet152,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}

# torch.backends.cudnn.benchmark = False
if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    )

    if args.gpu:
        net.to(device)
        #不使用分布式训练
        # net = torch.nn.DataParallel(
        #     net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    opt_params = net.parameters()
    #设置optimier
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    #学习率schduler
    if args.scheduler=="step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.first_milestone, args.second_milestone],
            gamma=0.1)
    elif args.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=300,
                                                               eta_min=1e-7)

    train_loader, val_loader = dataset_loader[
        args.dataset].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            val_size=0.1,
            val_seed=args.seed,
            pin_memory=args.gpu,
            contrastive = args.contrastive
        )

    # Creating summary writer in tensorboard
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff,
                                args.seed, args.contrastive)

    save_loc = f"{args.save_loc}/runs{args.runs}/{time_str}/"
    if args.ls:
        save_loc = f"{args.save_loc}/runs{args.runs}/{time_str}_labelsmooth/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    writer = SummaryWriter(f"{args.log_loc}/{args.runs}/{save_name}/{time_str}")
    print("Model save name", save_name)

    best_acc = 0
    is_best = False
    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss, train_acc = train_single_epoch(
            epoch,
            net,
            train_loader,
            optimizer,
            device,
            args.contrastive,
            label_smooth=args.ls,
            loss_mean=args.loss_mean,
        )

        val_acc = test_single_epoch(net, val_loader, device)

        writer.add_scalar("train_loss", train_loss, (epoch + 1))
        writer.add_scalar("train_acc", train_acc, (epoch + 1))
        writer.add_scalar("val_acc", val_acc, (epoch + 1))

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True
        if is_best:
            save_path = save_loc + save_name + "_best" + ".model"
            torch.save(net.state_dict(), save_path)
            print("Model saved to ", save_path)

    writer.close()
