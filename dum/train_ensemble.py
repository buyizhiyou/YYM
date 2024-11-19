"""
Script for training deep ensemble models.
"""

import argparse
import datetime
import os

import torch
import torch.backends.cudnn as cudnn
from torch import optim
# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
# Import network models
from net.resnet import resnet50
from net.vgg import vgg16
from net.wide_resnet import wrn
# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats_ensemble
from utils.train_utils import (model_save_name, save_config_file,
                               test_single_epoch, train_single_epoch)

dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}

dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn}

models = {"resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16}


def parseArgs():
    ensemble = 5
    parser = training_args()
    parser.add_argument(
        "--ensemble",
        type=int,
        default=ensemble,
        dest="ensemble",
        help="Number of ensembles to train",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net_ensemble = [
        models[args.model](
            spectral_normalization=args.sn,
            mod=args.mod,
            coeff=args.coeff,
            num_classes=num_classes,
        ).to(device) for _ in range(args.ensemble)
    ]

    optimizers = []
    schedulers = []
    train_loaders = []
    val_loaders = []

    for i, model in enumerate(net_ensemble):
        opt_params = model.parameters()
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
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.first_milestone, args.second_milestone],
            gamma=0.1,
        )
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(root=args.dataset_root,
                                                                                       batch_size=args.train_batch_size,
                                                                                       augment=args.data_aug,
                                                                                       val_size=0.1,
                                                                                       val_seed=args.seed,
                                                                                       pin_memory=args.gpu,
                                                                                       contrastive=args.contrastive)

        optimizers.append(optimizer)
        schedulers.append(scheduler)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive)
    print("Model save name", save_name)
    if args.ls:
        save_loc = f"{args.save_loc}/run{args.run}/ensemble/{save_name}/{time_str}_labelsmooth/"
    else:
        save_loc = f"{args.save_loc}/run{args.run}/ensemble/{save_name}/{time_str}/"  # 保存模型路径
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    log_loc = f"{args.log_loc}/run{args.run}/{save_name}/{time_str}"
    writer = SummaryWriter(log_loc)
    save_config_file(save_loc, args)

    best_acc = [0] * (args.ensemble)
    for epoch in range(0, args.epoch):
        for i, model in enumerate(net_ensemble):
            print("Ensemble Model: " + str(i))
            train_loss, train_acc = train_single_epoch(
                epoch,
                model,
                train_loaders[i],
                optimizers[i],
                device,
                loss_mean=args.loss_mean,
            )
            schedulers[i].step()

            if (epoch % 3 == 0):
                val_acc = test_single_epoch(epoch, model, val_loaders[i], device)

            if val_acc > best_acc[i]:
                best_acc[i] = val_acc
                save_path = save_loc + save_name + f"_{i}" + "_best" + ".model"
                torch.save(model.state_dict(), save_path)
                print("Model saved to ", save_path)

        writer.add_scalar(args.model + "_ensemble_" + "train_loss", train_loss, (epoch + 1))

    writer.close()
