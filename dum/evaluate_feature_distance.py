"""
Script to evaluate a single model.
"""
import os
import json
import math
import sys
import time
import numpy as np
import torch
import pickle as pkl
import glob, re
import argparse
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from sklearn.decomposition import PCA

# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.gauss as gauss
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet
import data_utils.ood_detection.fer2013 as fer2013
import data_utils.ood_detection.dtd as dtd
import data_utils.ood_detection.stl as stl
import data_utils.ood_detection.caltech256 as caltech256
import data_utils.ood_detection.place365 as place365

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50  ##这是现在的resnet，额外的加一层fc,测试run37之后都要用这个
# from net.resnet3 import resnet18, resnet50#这是以前的resnet，没有额外的加一层fc,测试run37之前都要用这个
from net.wide_resnet import wrn
from net.vgg import vgg16
from net.vit import vit

# Import metrics to compute
from metrics.feature_metrics import evaluate_class_separability, intra_class_distance, intra_class_variance, davies_bouldin_index, inter_class_distance, fisher_ratio, silhouette_score

# Import GMM utils
from utils.gmm_utils import get_embeddings
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name, seed_torch
from utils.args import eval_args

# Dataset params
dataset_num_classes = {"mnist": 10, "cifar10": 10, "cifar100": 100, "svhn": 10, "lsun": 10, "tiny_iamgenet": 200}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "fer2013": fer2013,
    "mnist": mnist,
    "lsun": lsun,
    "dtd": dtd,
    "stl": stl,
    "place365": place365,
    "caltech256": caltech256,
    "tiny_imagenet": tiny_imagenet
}

# Mapping model name to model function
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":

    seed_torch()
    args = eval_args().parse_args()
    # Checking if GPU is available
    cuda = torch.cuda.is_available()
    size = args.size
    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}
    model_to_num_dim = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048, "wide_resnet": 640, "vgg16": 512, "vit": 768}

    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    # Evaluating the models
    Intra_class_Distances = []
    # Intra_class_Variances = []
    Inter_class_Distances = []
    Davies_Bouldin_Indexs = []
    Silhouette_Coefficients = []
    Fisher_Ratios = []

    topt = None
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive)
    model_name = model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive) + "_best.model"

    model_files = sorted(glob.glob(f"{args.load_loc}/run{args.run}/{save_name}/*/{model_name}"))

    if len(model_files) == 0:
        print("no model files in current config")
        exit()

    for i, saved_model_name in enumerate(model_files):
        print(f"Run {args.run},OOD dataset {args.ood_dataset} Evaluating for {i}/{len(model_files)}: {saved_model_name}")
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.batch_size,
            size=size,
            augment=args.data_aug,  #False
            val_seed=(args.seed),
            val_size=0.0,  #这里0.1改为0，使用全部训练数据
            pin_memory=args.gpu,
        )
        #load model
        print(f"load {saved_model_name}")
        net = models[args.model](
            spectral_normalization=args.sn,
            mod=args.mod,
            num_classes=num_classes,
            temp=1.0,
        )
        net.to(device)
        net.load_state_dict(torch.load(str(saved_model_name), map_location=device), strict=True)
        net.eval()

        load_cache = True
        cache_path = saved_model_name.replace(".model", ".cache")
        if load_cache and os.path.exists(cache_path):
            print(f"load cache from {cache_path}")
            with open(cache_path, 'rb') as file:
                cache = pkl.load(file)
                embeddings = cache["embeddings"].to(device)
                labels = cache["labels"].to(device)
                norm_threshold = cache["norm_threshold"]
        else:
            embeddings, labels, norm_threshold = get_embeddings(
                net,
                train_loader,
                num_dim=model_to_num_dim[args.model],
                dtype=torch.double,
                device=device,
                storage_device=device,
            )
            cache = {"embeddings": embeddings.cpu(), "labels": labels.cpu(), "norm_threshold": norm_threshold}
            with open(cache_path, "wb") as f:
                pkl.dump(cache, f)

        features = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        Intra_class_Distance = intra_class_distance(features, labels)
        Davies_Bouldin_Index = davies_bouldin_index(features, labels)
        Inter_class_Distance = inter_class_distance(features, labels)
        Fisher_Ratio = fisher_ratio(features, labels)
        Silhouette_Coefficient = silhouette_score(features, labels)

        print(
            f"Intra_class_Distance:{Intra_class_Distance},Davies_Bouldin_Index:{Davies_Bouldin_Index},Inter_class_Distance:{Inter_class_Distance},Fisher_Ratio:{Fisher_Ratio},Silhouette_Coefficient:{Silhouette_Coefficient}"
        )
        Intra_class_Distances.append(Intra_class_Distance)
        Inter_class_Distances.append(Inter_class_Distance)
        Fisher_Ratios.append(Fisher_Ratio)
        Davies_Bouldin_Indexs.append(Davies_Bouldin_Index)
        Silhouette_Coefficients.append(Silhouette_Coefficient)
      
      
    res_dict = {}
    res_dict["mean"] = {}
    res_dict["values"] = {}
    
    res_dict["values"]["Inter_class_Distances"] = Inter_class_Distances
    res_dict["values"]["Intra_class_Distances"] = Intra_class_Distances
    res_dict["values"]["Fisher_Ratios"] = Fisher_Ratios
    res_dict["values"]["Davies_Bouldin_Indexs"] = Davies_Bouldin_Indexs
    res_dict["values"]["Silhouette_Coefficients"] = Silhouette_Coefficients
    
    res_dict["mean"]["Intra_class_Distances"] = round(np.mean(Intra_class_Distances),4)
    res_dict["mean"]["Davies_Bouldin_Indexs"] = round(np.mean(Davies_Bouldin_Indexs),4)
    res_dict["mean"]["Inter_class_Distances"] = round(np.mean(Inter_class_Distances),4)
    res_dict["mean"]["Fisher_Ratios"] = round(np.mean(Fisher_Ratios),4)
    res_dict["mean"]["Silhouette_Coefficients"] = round(np.mean(Silhouette_Coefficients),4)

    res_dict["info"] = vars(args)
    res_dict["files"] = model_files
    res_dict["annotations"] = "Intra_class_Distances/Davies_Bouldin_Indexs the lower the better，Inter_class_Distances/Fisher_Ratios/Silhouette_Coefficients the higher the better"

    saved_name = "res_" + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed,
                                          args.contrastive) + "_" + args.dataset + "_" + "_" + args.perturbation + "_feature.json"
    saved_dir = f"./results/run{args.run}/"
    if (not os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    with open(os.path.join(saved_dir, saved_name), "w") as f:
        json.dump(res_dict, f)
        print(f"save to {os.path.join(saved_dir,saved_name)}")
