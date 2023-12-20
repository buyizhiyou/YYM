#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_bayesian.py
@Time    :   2023/11/17 13:35:58
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

import argparse
import datetime
import glob
import os
import time
import warnings
import numpy as np 

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import warmup_scheduler
import yaml
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from data_utils.get_datasets import get_dataset
from model_utils.get_models import get_model
from utils.loss import LabelSmoothingCrossEntropyLoss
from utils.metrics import accuracy
from utils.misc import argsdict, seed_torch
from utils.randomaug import RandAugment,MixUp,CutMix
from utils.visual import AverageMeter, ProgressMeter, Summary

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config', help='yaml config file')

best_acc1 = 0
def main():
    configargs = parser.parse_args()
    with open(configargs.config, "r") as f:
        cfg = yaml.safe_load(f)
    args = argsdict(cfg)  # 包装字典，可以通过.访问

    # if args.seed is not None:
    #     print("set random seed")
    #     seed_torch(args.seed)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # 是否开启多卡训练

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    model = get_model(args.arch, 10, args.use_torchvision,args.pretrained,args.use_bayesian)
    # summary(model,(3, 32, 32),device="cpu")

    # translate deterministic network into bayesian network
    moped_enable = False
    if len(args.moped_init_model) > 0:  # use moped method if trained dnn model weights are provided
        moped_enable = True
    const_bnn_prior_parameters = {
        "prior_mu": args.prior_mu,
        "prior_sigma": args.prior_sigma,
        "posterior_mu_init": args.posterior_mu_init,
        "posterior_rho_init": args.bnn_rho_init,
        "type": "Flipout" if args.use_flipout_layers else "Reparameterization",## Flipout or Reparameterization
        "moped_enable": moped_enable,  # initialize mu/sigma from the dnn weights
        "moped_delta": args.moped_delta_factor,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)

    # ddp并行训练配置
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(
                    (args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
                print("using ddp mode")
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 定义 loss function (criterion), optimizer, and learning rate scheduler
    if args.labelsmoothing:
        criterion = LabelSmoothingCrossEntropyLoss(
            args.num_classes, smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(
        ), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay) #训练bnn时weight_decay置0
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)
    elif args.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)
    #使用warmup策略
    if args.warmup:
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1.,
                                                            total_epoch=5, after_scheduler=scheduler)

    curr_time = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
    time_str = "2023_11_29_11_13_47"
    model_dir = f"{args.saved_models}/{args.mode}/{args.arch}/{time_str}"
    log_dir = f"{args.logs}/{args.mode}/{args.arch}/{time_str}"
    try:
        os.makedirs(model_dir)
        os.makedirs(log_dir)
    except:
        pass

    # 模型恢复
    if args.resume:
        reg_path = os.path.join(model_dir, f"{args.arch}_best_model*.pth")
        files_path = glob.glob(reg_path)
        if files_path:
            resume_path = files_path[0]
            print("=> loading checkpoint '{}'".format(resume_path))
            if args.gpu is None:
                checkpoint = torch.load(resume_path)
            elif torch.cuda.is_available():
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(resume_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            print(
                f"resume model start epoch {args.start_epoch},best acc1:{best_acc1}")
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1)
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if args.warmup:
                scheduler.after_scheduler.optimizer = optimizer
            else:
                scheduler.optimizer = optimizer
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_dir))

    # 数据集加载
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.RandomCrop(args.size, padding=4),
            transforms.RandomGrayscale(),  # add
            transforms.GaussianBlur(3),  # add
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),  # pytorch doc std
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ]
    )
    # 添加额外的数据增强
    if args.aug:
        print("add more augmentation")
        # train_transform.transforms.insert(0, RandAugment(N, p)) #自己实现的autoaugmentation,不如使用下面的
        auto_aug =  transforms.AutoAugment(policy=transforms.AutoAugmentPolicy('cifar10'), 
                                            interpolation=transforms.InterpolationMode.BILINEAR)#torchvision里的autoaugmentation
        train_transform.transforms.insert(1,auto_aug)

    train_dataset, val_dataset = get_dataset(
        args.data, "./data", train_transform, val_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)  
    # 在测试集上评估模型
    if args.evaluate:
        validate(val_loader, model, args)
        return

    writer = SummaryWriter(log_dir)
    with open("logs/model_parameters_map.yaml", "a") as f:  # 保存模型日期和训练参数
        yaml.dump({f"{args.arch}_{args.mode}_{time_str}": dict(args)}, f)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            # 在每个周期开始之前，可以调用train_sampler.set_epoch(epoch)来使得数据打的更乱。
            train_sampler.set_epoch(epoch)

        # train for one epoch
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion,
              optimizer, writer, epoch, device, args)
        scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, model, args)
        writer.add_scalar("acc1/test", acc1, epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # 模型保存
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "best_acc1": best_acc1
            }

            if is_best:
                # 删除旧模型文件
                files = glob.glob(os.path.join(
                    model_dir, f"{args.arch}_best_model_*.pth"))
                for f in files:
                    os.remove(f)
                # 保存准确率最高的模型文件
                torch.save(state, os.path.join(
                    model_dir, f"{args.arch}_best_model_{best_acc1:.2f}.pth"))


def train(train_loader, model, criterion, optimizer, writer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    if args.use_cutmix:
        cutmix = CutMix(args.size, beta=1.)
    if args.use_mixup:
        mixup = MixUp(alpha=1.)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
            
        with torch.autocast("cuda", enabled=args.use_amp):
            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output = model(images)
                kl = get_kl_loss(model)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            if args.use_cutmix or args.use_mixup:
                if args.use_cutmix:
                    images, label, rand_label, lambda_= cutmix((images, target))
                elif args.use_mixup:
                    if np.random.rand() <= 0.8:
                        images, label, rand_label, lambda_ = mixup((images, target))
                    else:
                        images, label, rand_label, lambda_ = images, label, torch.zeros_like(label), 1.
                output = model(images)
                cross_entropy_loss = criterion(output, label)*lambda_ + criterion(output, rand_label)*(1.-lambda_)
            else:
                output = model(images)
                cross_entropy_loss = criterion(output, target)

            scaled_kl = kl / args.batch_size
            # ELBO loss
            loss = cross_entropy_loss + scaled_kl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i + 1)

    # log loss and acc1
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank == 0):
        writer.add_scalar("Loss/train", losses.avg, epoch)
        writer.add_scalar("acc1/train", top1.avg, epoch)
        writer.flush()


def validate(val_loader, model, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                output_mc = []
                for _ in range(args.num_monte_carlo):
                    output = torch.softmax(model.forward(images),dim=1)#输出的概率
                    output_mc.append(output)
                output_ = torch.stack(output_mc,dim=0)#NumMCxBatchSizexNum_classes

                # measure accuracy and record loss
                acc1= accuracy(torch.mean(output_, dim=0), target, topk=(1, ))[0]
                top1.update(acc1[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler)
                                                 * args.world_size < len(val_loader.dataset))),
        [batch_time,  top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    main()
