"""
Script to evaluate a single model.
"""
import argparse
import glob
import json
import math
import os
import pickle as pkl
import re

import torch
import torch.backends.cudnn as cudnn

import data_utils.ood_detection.caltech256 as caltech256
# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.dtd as dtd
import data_utils.ood_detection.fer2013 as fer2013
import data_utils.ood_detection.gauss as gauss
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.place365 as place365
import data_utils.ood_detection.stl as stl
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet
from metrics.calibration_metrics import expected_calibration_error
# Import metrics to compute
from metrics.classification_metrics import (test_classification_net,
                                            test_classification_net_ensemble,
                                            test_classification_net_logits)
from metrics.ood_metrics import (get_roc_auc, get_roc_auc_ensemble,
                                 get_roc_auc_ensemble_adv, get_roc_auc_logits)
from metrics.uncertainty_confidence import (confidence, entropy, logsumexp,
                                            maxval, sumexp)
# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.vgg import vgg16
from net.vit import vit
# from net.resnet2 import resnet18, resnet50
from net.wide_resnet import wrn
from utils.args import eval_args
from utils.ensemble_utils import ensemble_forward_pass, load_ensemble
from utils.eval_utils import model_load_name
# Import GMM utils
from utils.gmm_utils import (get_embeddings, gmm_evaluate,
                             gmm_evaluate_for_adv,
                             gmm_evaluate_with_perturbation,
                             gmm_evaluate_with_perturbation_for_adv, gmm_fit,
                             gradient_norm_collect, maxp_evaluate,
                             maxp_evaluate_with_perturbation)
from utils.kde_utils import kde_evaluate, kde_fit
# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature
from utils.train_utils import model_save_name

# Dataset params
dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "lsun": 10, "tiny_iamgenet": 200}

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
models = {"lenet": lenet, "resnet18": resnet18, "resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16, "vit": vit}

model_to_num_dim = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048, "wide_resnet": 640, "vgg16": 512, "vit": 768}

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":

    args = eval_args().parse_args()
    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")

    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    # Evaluating the models
    accuracies = []

    # Pre temperature scaling
    eces = []
    t_eces = []
    # m1_fpr95s = []
    m1_aurocs = []
    m1_auprcs = []
    # m2_fpr95s = []
    m2_aurocs = []
    m2_auprcs = []
    epsilons = []

    topt = None
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive)
    model_name = model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive) + "_best.model"

    if args.evaltype == "ensemble":
        model_files = sorted(glob.glob(f"{args.load_loc}/run{args.run}/ensemble/{save_name}/*"))
    else:
        model_files = sorted(glob.glob(f"{args.load_loc}/run{args.run}/{save_name}/*/{model_name}"))

    if len(model_files) == 0:
        print("No models checkpoint files")
        exit()
        
    ##加载模型文件
    for i, saved_model_name in enumerate(model_files):
        print(f"Run {args.run},inD dataset {args.dataset},OOD dataset {args.ood_dataset} Evaluating for {i}/{len(model_files)}: {saved_model_name}")
        if args.evaltype == "ensemble":
            #load ensemble models
            print(f"load {saved_model_name}")
            net_ensemble = load_ensemble(ensemble_loc=saved_model_name,
                                         model_name=args.model,
                                         device=device,
                                         num_classes=num_classes,
                                         spectral_normalization=args.sn,
                                         mod=args.mod,
                                         coeff=args.coeff,
                                         seed=(5 * i + 1))
        else:
            #load dataset
            train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                root=args.dataset_root,
                batch_size=args.batch_size,
                augment=args.data_aug,  #False
                val_seed=(args.seed),
                val_size=0.1,
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
            if args.gpu:
                net.to(device)
                
            net.load_state_dict(torch.load(str(saved_model_name), map_location=device), strict=True)
            net.eval()

        if args.evaltype == "ensemble":
            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=512, pin_memory=args.gpu,size=32,sample_size=200)

            (
                conf_matrix,
                accuracy,
                labels_list,
                predictions,
                confidences,
            ) = test_classification_net_ensemble(net_ensemble, test_loader, device)
            print(f"{saved_model_name} accu:{accuracy}")
            # ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

            (_, _, _), (_, _, _), m1_auroc_adv, m1_auprc_adv = get_roc_auc_ensemble_adv(net_ensemble, test_loader, "mutual_information", device)
            (_, _, _), (_, _, _), m2_auroc_adv, m2_auprc_adv = get_roc_auc_ensemble_adv(net_ensemble, test_loader, "entropy", device)
            print(f"ensemble,mutual_info:m1_auroc1:{m1_auroc_adv:.4f},m1_auprc:{m1_auprc_adv:.4f}; entropy:m2_auroc:{m2_auroc_adv:.4f},m2_auprc:{m2_auprc_adv:.4f}")
            epsilon=0
        else:
            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=200, pin_memory=args.gpu,size=224,sample_size=200)
            (
                conf_matrix,
                accuracy,
                labels_list,
                predictions,
                confidences,
            ) = test_classification_net(net, test_loader, device)

            if (args.evaltype == "gmm"):
                # Evaluate a GMM model
                print("Evaluate gmm Model")
                cache_path = re.sub(r"[^/]*_best.model", "cache", saved_model_name)

                if os.path.exists(cache_path):
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

                try:
                    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                    logits, _, _ = gmm_evaluate(
                        net,
                        gaussians_model,
                        test_loader,
                        device=device,
                        num_classes=num_classes,
                        storage_device=device,
                    )

                    test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=1, pin_memory=args.gpu,size=224,sample_size=200)
                    logits_adv, _, _ = gmm_evaluate_for_adv(
                        net,
                        gaussians_model,
                        test_loader,
                        device=device,
                        num_classes=num_classes,
                        storage_device=device,
                        perturbation=args.perturbation
                    )

                    m2_res = []
                    for epsilon in [-0.0001,-0.0005,-0.001,-0.005,-0.01,-0.05,-0.1]:
                    # for epsilon in [-0.05,-0.1]:
                        for temp in [1]:
                            if args.perturbation in ["cw", "bim", "fgsm", "pgd"]:
                                print(f"add noise:{args.perturbation}")
                                #sample_size越小，结果越大
                                test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root, batch_size=1, pin_memory=args.gpu,size=224,sample_size=200)
                                logits_adv2, _, _, acc, acc_perturb = gmm_evaluate_with_perturbation_for_adv(
                                    net,
                                    gaussians_model,
                                    test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    storage_device=device,
                                    norm_threshold=norm_threshold,
                                    perturbation=args.perturbation,
                                    epsilon=epsilon,
                                    temperature=temp,
                                )


                            _, m1_auroc_adv, m1_auprc_adv = get_roc_auc_logits(logits, logits_adv, maxval, device, conf=True)
                            _, m2_auroc_adv, m2_auprc_adv = get_roc_auc_logits(logits, logits_adv2, maxval, device, conf=True)
                            print(
                                f"{args.model},adv:epsilon:{epsilon},m1_auroc_adv:{m1_auprc_adv:.4f},m1_auprc_adv:{m1_auprc_adv:.4f},m2_auroc_adv:{m2_auroc_adv:.4f},m2_auprc_adv:{m2_auprc_adv:.4f}"
                            )

                            m2_res.append([m2_auroc_adv, m2_auprc_adv, epsilon])
                    m2_auroc_adv, m2_auprc_adv, epsilon = sorted(m2_res)[-1]  #从小到大排序，并且取最大的

                    print(f"{args.model},adv:epsilon:{epsilon},m1_auroc_adv:{m1_auprc_adv:.4f},m1_auprc_adv:{m1_auprc_adv:.4f},m2_auroc_adv:{m2_auroc_adv:.4f},m2_auprc_adv:{m2_auprc_adv:.4f}")
                    
                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
                    continue
           
        epsilons.append(epsilon)
        accuracies.append(accuracy)
        eces.append(0.0)
        t_eces.append(0.0)
        # eces.append(ece)
        # t_eces.append(t_ece)
        m1_aurocs.append(m1_auroc_adv)
        m1_auprcs.append(m1_auprc_adv)
        m2_aurocs.append(m2_auroc_adv)
        m2_auprcs.append(m2_auprc_adv)


    ##保存结果
    accuracy_tensor = torch.tensor(accuracies)
    ece_tensor = torch.tensor(eces)
    t_ece_tensor = torch.tensor(t_eces)
    m1_auroc_tensor = torch.tensor(m1_aurocs)
    m1_auprc_tensor = torch.tensor(m1_auprcs)
    m2_auroc_tensor = torch.tensor(m2_aurocs)
    m2_auprc_tensor = torch.tensor(m2_auprcs)

    mean_accuracy = torch.mean(accuracy_tensor)
    mean_ece = torch.mean(ece_tensor)
    mean_t_ece = torch.mean(t_ece_tensor)
    mean_m1_auroc = torch.mean(m1_auroc_tensor)
    mean_m1_auprc = torch.mean(m1_auprc_tensor)
    mean_m2_auroc = torch.mean(m2_auroc_tensor)
    mean_m2_auprc = torch.mean(m2_auprc_tensor)

    std_accuracy = torch.std(accuracy_tensor) / math.sqrt(accuracy_tensor.shape[0])
    std_ece = torch.std(ece_tensor) / math.sqrt(ece_tensor.shape[0])
    std_t_ece = torch.std(t_ece_tensor) / math.sqrt(t_ece_tensor.shape[0])
    std_m1_auroc = torch.std(m1_auroc_tensor) / math.sqrt(m1_auroc_tensor.shape[0])
    std_m1_auprc = torch.std(m1_auprc_tensor) / math.sqrt(m1_auprc_tensor.shape[0])
    std_m2_auroc = torch.std(m2_auroc_tensor) / math.sqrt(m2_auroc_tensor.shape[0])
    std_m2_auprc = torch.std(m2_auprc_tensor) / math.sqrt(m2_auprc_tensor.shape[0])

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = round(mean_accuracy.item(), 4)
    res_dict["mean"]["ece"] = round(mean_ece.item(), 4)
    res_dict["mean"]["t_ece"] = round(mean_t_ece.item(), 4)
    res_dict["mean"]["m1_auroc"] = round(mean_m1_auroc.item(), 4)
    res_dict["mean"]["m1_auprc"] = round(mean_m1_auprc.item(), 4)
    res_dict["mean"]["m2_auroc"] = round(mean_m2_auroc.item(), 4)
    res_dict["mean"]["m2_auprc"] = round(mean_m2_auprc.item(), 4)

    res_dict["std"] = {}
    res_dict["std"]["accuracy"] = std_accuracy.item()
    res_dict["std"]["ece"] = std_ece.item()
    res_dict["std"]["t_ece"] = std_t_ece.item()
    res_dict["std"]["m1_auroc"] = std_m1_auroc.item()
    res_dict["std"]["m1_auprc"] = std_m1_auprc.item()
    res_dict["std"]["m2_auroc"] = std_m2_auroc.item()
    res_dict["std"]["m2_auprc"] = std_m2_auprc.item()

    res_dict["values"] = {}
    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["t_ece"] = t_eces
    res_dict["values"]["m1_auroc"] = m1_aurocs
    res_dict["values"]["m1_auprc"] = m1_auprcs
    res_dict["values"]["m2_auroc"] = m2_aurocs
    res_dict["values"]["m2_auprc"] = m2_auprcs

    res_dict["info"] = vars(args)
    res_dict["files"] = model_files
    res_dict["epsilon"] = epsilons
    res_dict["annotations"] = "adv detection,m1:without noise,m2:add noise"

    saved_name = "adv_res_" + model_save_name(
        args.model, args.sn, args.mod, args.coeff, args.seed,
        args.contrastive) + "_" + args.evaltype + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.perturbation + ".json"
    saved_dir = f"./results/run{args.run}/"
    if (not os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    with open(os.path.join(saved_dir, saved_name), "w") as f:
        json.dump(res_dict, f)
        print(f"save to {os.path.join(saved_dir,saved_name)}")
