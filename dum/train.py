"""
Script for training a single model for OOD detection.
"""

import argparse
import datetime
import time
import json
import os

import torch
import torch.backends.cudnn as cudnn

from net.lenet import lenet
from net.resnet import resnet18, resnet50  #自己实现的spectral norm
# from net.resnet2 import resnet18, resnet50 #官方实现的spectral norm
from net.vgg import vgg16  #自己实现的
# from net.vgg2 import vgg16 #官方实现的
from net.wide_resnet import wrn
from net.vit import vit
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import data_utils.dirty_mnist as dirty_mnist

import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn

from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import (model_save_name, save_config_file, test_single_epoch, train_single_epoch)

dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
}

models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}

# torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    # print(os.environ)
    # rank = int(os.environ['RANK'])
    # local_rank = int(os.environ['LOCAL_RANK'])
    # # local_rank = args.local_rank
    # master_addr = os.environ['MASTER_ADDR']
    # master_port = os.environ['MASTER_PORT']
    # world_size = int(os.environ['WORLD_SIZE'])
    # print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    # dist.init_process_group(backend="nccl", init_method='env://')

    torch.manual_seed(0)
    # device = torch.device(f"cuda:{args.gpu}")

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](spectral_normalization=args.sn, mod=args.mod, num_classes=num_classes).to(device)
    # net = DDP(net, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    if args.gpu:
        net.to(device)
        cudnn.benchmark = True

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
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            # milestones=[args.first_milestone, args.second_milestone],
            milestones=[0.3 * args.epoch, 0.6 * args.epoch, 0.9 * args.epoch],
            gamma=0.1,
            verbose=False)
    elif args.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-7, verbose=False)

    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(root=args.dataset_root,
                                                                                   batch_size=args.train_batch_size,
                                                                                   augment=args.data_aug,
                                                                                   val_size=0.1,
                                                                                   val_seed=args.seed,
                                                                                   pin_memory=args.gpu,
                                                                                   contrastive=args.contrastive)

    test_loader = dataset_loader[args.dataset].get_test_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
    )

    # Creating summary writer in tensorboard
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive)
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
    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss, train_acc = train_single_epoch(
            epoch,
            net,
            train_loader,
            optimizer,
            device,
            args.contrastive,
            adv=args.adv,
            label_smooth=args.ls,
            loss_mean=args.loss_mean,
        )

        if epoch % 3 == 0:
            val_acc = test_single_epoch(epoch, net, val_loader, device)
            writer.add_scalar("train_loss", train_loss, (epoch + 1))
            writer.add_scalar("train_acc", train_acc, (epoch + 1))
            writer.add_scalar("val_acc", val_acc, (epoch + 1))
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=(epoch + 1))

        scheduler.step()

        if epoch == 350:  # 训练完第一阶段
            save_path = save_loc + save_name + "_mid" + ".model"
            torch.save(net.state_dict(), save_path)
            print("Model saved to ", save_path)

        if val_acc > best_acc:
            best_acc = val_acc

            save_path = save_loc + save_name + "_best" + ".model"
            torch.save(net.state_dict(), save_path)
            print("Model saved to ", save_path)

    writer.close()
