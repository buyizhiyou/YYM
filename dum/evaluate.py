"""
Script to evaluate a single model.
"""
import os
import json
import math
import shutil
import sys
import time
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
from metrics.classification_metrics import (test_classification_net, test_classification_net_logits, test_classification_net_ensemble)
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp, confidence, sumexp, maxval
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble, auroc, auprc

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit, maxp_evaluate, gradient_norm_collect, gmm_evaluate_for_adv, gmm_evaluate_with_perturbation_for_adv, gmm_evaluate_with_perturbation, maxp_evaluate_with_perturbation, gmm_evaluate_with_perturbation2, gmm_evaluate_with_perturbation3
from utils.kde_utils import kde_evaluate, kde_fit
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name, seed_torch
from utils.args import eval_args
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass

# # Temperature scaling
# from utils.temperature_scaling import ModelWithTemperature

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
    accuracies = []

    # Pre temperature scaling
    eces = []
    t_eces = []
    m1_aurocs = []
    m1_auprcs = []
    m2_aurocs = []
    m2_auprcs = []
    epsilons = []
    ece = 0.0
    t_ece = 1.0
    norm_threshold = 0.0
    pca_res = {}

    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive)
    model_name = model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, args.contrastive) + "_best.model"

    if args.evaltype == "ensemble":
        model_files = sorted(glob.glob(f"{args.load_loc}/run{args.run}/ensemble/{save_name}/*"))
    else:
        model_files = sorted(glob.glob(f"{args.load_loc}/run{args.run}/{save_name}/*/{model_name}"))

    if len(model_files) == 0:
        print("no model files in current config")
        exit()

    for i, saved_model_name in enumerate(model_files):
        print(f"Run {args.run},OOD dataset {args.ood_dataset} Evaluating for {i}/{len(model_files)}: {saved_model_name}")
        if args.evaltype == "ensemble":
            val_loaders = []
            for j in range(args.ensemble):
                train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                    root=args.dataset_root,
                    batch_size=args.batch_size,
                    size=size,
                    augment=args.data_aug,
                    val_seed=(args.seed + (5 * i) + j),
                    val_size=0.1,  #这里0.1改为0，使用全部训练数据
                    pin_memory=args.gpu,
                )
                val_loaders.append(val_loader)
            # Evaluate an ensemble
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

        if args.evaltype == "ensemble":
            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                       batch_size=args.batch_size,
                                                                       size=size,
                                                                       pin_memory=args.gpu)
            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                               batch_size=args.batch_size,
                                                                               size=size,
                                                                               pin_memory=args.gpu)
            (
                conf_matrix,
                accuracy,
                labels_list,
                predictions,
                confidences,
            ) = test_classification_net_ensemble(net_ensemble, test_loader, device)
            print(f"{saved_model_name} accu:{accuracy}")
            # ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

            (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_ensemble(net_ensemble, test_loader, ood_test_loader, "mutual_information", device)
            (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_ensemble(net_ensemble, test_loader, ood_test_loader, "entropy", device)
            print(f"mutual_info:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f}; entropy:m2_auroc:{m2_auroc:.4f},m2_auprc:{m2_auprc:.4f}")
            # Temperature scale the ensemble
            # t_ensemble = []
            # for model, val_loader in zip(net_ensemble, val_loaders):
            #     t_model = ModelWithTemperature(model, device)
            #     t_model.set_temperature(val_loader)
            #     t_ensemble.append(t_model)

            # (
            #     t_conf_matrix,
            #     t_accuracy,
            #     t_labels_list,
            #     t_predictions,
            #     t_confidences,
            # ) = test_classification_net_ensemble(t_ensemble, test_loader, device)
            # t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)
            epsilon = 0.0
        else:
            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                       batch_size=args.batch_size,
                                                                       size=size,
                                                                       pin_memory=args.gpu)
            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                               batch_size=args.batch_size,
                                                                               size=size,
                                                                               pin_memory=args.gpu)
            (
                conf_matrix,
                accuracy,
                labels_list,
                predictions,
                confidences,
            ) = test_classification_net(net, test_loader, device)
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

            # #校准，调整温度系数
            # temp_net = ModelWithTemperature(net, device)
            # temp_net.set_temperature(val_loader)
            # net.temp = temp_net.temperature

            # (
            #     t_conf_matrix,
            #     t_accuracy,
            #     t_labels_list,
            #     t_predictions,
            #     t_confidences,
            # ) = test_classification_net(temp_net, test_loader, device)
            # t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

            if (args.evaltype == "gmm"):
                # Evaluate a GMM model
                print("GMM Model")
                cache_path = saved_model_name.replace(".model", ".cache")
                param_path = saved_model_name.replace(".model", "_gmm2.model")
                use_param = False
                if use_param and os.path.exists(param_path):
                    params = torch.load(param_path, map_location=device)
                    classwise_mean_features = params['mu'].to(device)
                    cholesky_factors = params['cholesky_factors'].to(device)
                    classwise_cov_features = torch.matmul(cholesky_factors, cholesky_factors.transpose(-1, -2)).to(device)
                    DOUBLE_INFO = torch.finfo(torch.double)
                    JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-30, 1, 1)]
                    for jitter_eps in JITTERS:
                        try:  #协方差矩阵要求正定,但是样本协方差矩阵不一定正定,所以加上对角阵转化为正定的
                            jitter = jitter_eps * torch.eye(
                                model_to_num_dim[args.model],
                                device=device,
                            ).unsqueeze(0)
                            gaussians_model = torch.distributions.MultivariateNormal(
                                loc=classwise_mean_features,
                                covariance_matrix=(classwise_cov_features + jitter),
                            )
                            print("jitter", jitter_eps)
                        except:
                            continue
                        break
                else:
                    load_cache = True
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

                    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)

                save_path = "gmm_params.pth"
                torch.save({'loc': gaussians_model.loc, 'covariance_matrix': gaussians_model.covariance_matrix}, save_path)
                storage = os.path.getsize(save_path)
                os.remove(save_path)

                pca = None
                start = time.time()
                logits, labels, preds = gmm_evaluate(
                    net,
                    gaussians_model,
                    pca,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                ood_logits, ood_labels, _ = gmm_evaluate(
                    net,
                    gaussians_model,
                    pca,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )
                interval1 = time.time() - start
                if args.perturbation == 'pca':
                    ece = interval1  #如果做pca的实验，ece暂存interval1
                m1_fpr95, m1_auroc, m1_auprc = get_roc_auc_logits(logits, ood_logits, maxval, device, conf=True)

                m2_res = []
                if args.perturbation in ["fgsm2", "fgsm"]:
                    eps = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.009, 0.01]
                elif args.perturbation in ["fgsm3"]:
                    eps = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
                elif args.perturbation == "misclassified":
                    eps = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
                elif args.perturbation == "adv":
                    eps = [0.0001, 0.001, 0.002, 0.003, 0.005, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
                else:
                    eps = [0.0001]

                for epsilon in eps:
                    for temp in [1]:
                        if args.perturbation in ["fgsm"]:
                            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                                       batch_size=1,
                                                                                       size=size,
                                                                                       pin_memory=args.gpu)
                            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                                               batch_size=1,
                                                                                               size=size,
                                                                                               pin_memory=args.gpu)

                            print(f"add noise:{args.perturbation}")
                            logits2, labels2, preds2, acc, acc_perturb = gmm_evaluate_with_perturbation(
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
                            inf = torch.min(logits2).item()
                            ood_logits2, ood_labels2, _, _, _ = gmm_evaluate_with_perturbation(
                                net,
                                gaussians_model,
                                ood_test_loader,
                                device=device,
                                num_classes=num_classes,
                                storage_device=device,
                                norm_threshold=norm_threshold,
                                perturbation=args.perturbation,
                                epsilon=epsilon,
                                temperature=temp,
                                inf=inf,
                            )
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device,
                                                                              conf=True)  #这里使用maxval是求最大logP，使用logsumexp是求平均logP
                            #TODO:这里也是一个改进，我们使用maxval,而不是logsumexp

                            print(
                                f"noise-:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};noise+:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation in ["fgsm2"]:
                            '''
                            扰动两次
                            '''
                            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                                       batch_size=1,
                                                                                       size=size,
                                                                                       pin_memory=args.gpu)
                            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                                               batch_size=1,
                                                                                               size=size,
                                                                                               pin_memory=args.gpu)

                            print(f"add noise:{args.perturbation}")
                            logits2, labels2, preds2, acc, acc_perturb = gmm_evaluate_with_perturbation2(
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
                            inf = torch.min(logits2).item()
                            ood_logits2, ood_labels2, _, _, _ = gmm_evaluate_with_perturbation2(
                                net,
                                gaussians_model,
                                ood_test_loader,
                                device=device,
                                num_classes=num_classes,
                                storage_device=device,
                                norm_threshold=norm_threshold,
                                perturbation=args.perturbation,
                                epsilon=epsilon,
                                temperature=temp,
                                inf=inf,
                            )
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device,
                                                                              conf=True)  #这里使用maxval是求最大logP，使用logsumexp是求平均logP
                            #TODO:这里也是一个改进，我们使用maxval,而不是logsumexp

                            print(
                                f"noise-:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};noise+:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation in ["fgsm3"]:
                            '''
                            加梯度改为减梯度
                            '''
                            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                                       batch_size=1,
                                                                                       size=size,
                                                                                       pin_memory=args.gpu)
                            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                                               batch_size=1,
                                                                                               size=size,
                                                                                               pin_memory=args.gpu)

                            print(f"add noise:{args.perturbation}")
                            logits2, labels2, preds2, acc, acc_perturb = gmm_evaluate_with_perturbation3(
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
                            inf = torch.min(logits2).item()
                            ood_logits2, ood_labels2, _, _, _ = gmm_evaluate_with_perturbation3(
                                net,
                                gaussians_model,
                                ood_test_loader,
                                device=device,
                                num_classes=num_classes,
                                storage_device=device,
                                norm_threshold=norm_threshold,
                                perturbation=args.perturbation,
                                epsilon=epsilon,
                                temperature=temp,
                                inf=inf,
                            )
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device,
                                                                              conf=True)  #这里使用maxval是求最大logP，使用logsumexp是求平均logP
                            #TODO:这里也是一个改进，我们使用maxval,而不是logsumexp

                            print(
                                f"noise-:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};noise+:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation == "gradnorm1":  #使用gradient norm,type=1:logP求梯度
                            print("using gradient norm")
                            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                                       batch_size=1,
                                                                                       size=size,
                                                                                       pin_memory=args.gpu)
                            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                                               batch_size=1,
                                                                                               size=size,
                                                                                               pin_memory=args.gpu)
                            logits2 = gradient_norm_collect(
                                net,
                                gaussians_model,
                                test_loader,
                                device=device,
                                storage_device=device,
                                norm=1,
                                type=1,
                            )
                            ood_logits2 = gradient_norm_collect(net,
                                                                gaussians_model,
                                                                ood_test_loader,
                                                                device=device,
                                                                storage_device=device,
                                                                norm=1,
                                                                type=1)
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, None, device, conf=True)

                            print(
                                f"ddu:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};gradNorm:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation == "gradnorm2":  #使用gradient norm，type=2,p(x)求梯度
                            print("using gradient norm")
                            test_loader = dataset_loader[args.dataset].get_test_loader(root=args.dataset_root,
                                                                                       batch_size=1,
                                                                                       size=size,
                                                                                       pin_memory=args.gpu)
                            ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(root=args.dataset_root,
                                                                                               batch_size=1,
                                                                                               size=size,
                                                                                               pin_memory=args.gpu)
                            logits2 = gradient_norm_collect(net, gaussians_model, test_loader, device=device, storage_device=device, norm=1, type=2)
                            ood_logits2 = gradient_norm_collect(net,
                                                                gaussians_model,
                                                                ood_test_loader,
                                                                device=device,
                                                                storage_device=device,
                                                                norm=1,
                                                                type=2)
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, None, device, conf=True)

                            print(
                                f"ddu:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};gradNorm:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation == "misclassified":  #识别误分类
                            logits2, labels2, outs2, acc, acc_perturb = gmm_evaluate_with_perturbation(
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
                            confidences2, preds2 = torch.max(F.softmax(outs2, dim=1), dim=1)

                            t_ece = expected_calibration_error(confidences2.cpu().numpy(), preds2.cpu().numpy(), labels2.cpu().numpy(), num_bins=15)
                            #logits,labels,preds
                            mis = (preds == labels).to(torch.int).cpu().numpy()
                            mis2 = (preds2 == labels2).to(torch.int).cpu().numpy()
                            uncertainty = maxval(logits).cpu().numpy()
                            uncertainty2 = maxval(logits2).cpu().numpy()  #对比logsumexp
                            m1_auroc = auroc(uncertainty, mis)
                            m1_auprc = auprc(uncertainty, mis)
                            m2_auroc = auroc(uncertainty2, mis2)
                            m2_auprc = auprc(uncertainty2, mis2)
                            print(f"m1_auroc_mis:{m1_auroc},m1_auprc_mis:{m1_auprc},m2_auroc_mis:{m2_auroc},m2_auprc_mis:{m2_auprc}")
                        elif args.perturbation == "adv":
                            #TODO:分析对抗样本
                            logits_adv, _, _ = gmm_evaluate_for_adv(
                                net,
                                gaussians_model,
                                test_loader,
                                device=device,
                                num_classes=num_classes,
                                storage_device=device,
                            )
                            logits_adv2, _, _, _, _ = gmm_evaluate_with_perturbation_for_adv(
                                net,
                                gaussians_model,
                                test_loader,
                                device=device,
                                num_classes=num_classes,
                                storage_device=device,
                            )
                            # _, m1_auroc, m1_auprc = get_roc_auc_logits(logits, logits_adv, logsumexp, device, conf=True)
                            # _, m2_auroc, m2_auprc = get_roc_auc_logits(logits, logits_adv2, logsumexp, device, conf=True)
                            _, m1_auroc, m1_auprc = get_roc_auc_logits(logits, logits_adv, maxval, device, conf=True)
                            _, m2_auroc, m2_auprc = get_roc_auc_logits(logits, logits_adv2, maxval, device, conf=True)
                            print(f"m1_auroc_adv:{m1_auprc},m1_auprc_adv:{m1_auprc};epsion:{epsilon},m2_auroc_adv:{m2_auroc},m2_auprc_adv:{m2_auprc}")
                        elif args.perturbation == "none":  #不使用扰动
                            logits2 = logits
                            ood_logits2 = ood_logits
                            m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device,
                                                                              conf=True)  #这里使用maxval是求最大logP，使用logsumexp是求平均logP
                            #TODO:这里也是一个改进，我们使用maxval,而不是logsumexp

                            print(
                                f"none,m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};noise+:epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_aupr:{m2_auprc:.4f}"
                            )
                        elif args.perturbation == "pca":  #pca降维
                            res = []
                            use_pca = True

                            if args.model == "resnet50":
                                pca_dimension = list(range(128, model_to_num_dim[args.model], 128))
                            else:
                                pca_dimension = list(range(128, model_to_num_dim[args.model], 32))
                            pca_storage = []
                            pca_interval = []
                            pca_auroc = []
                            pca_auprc = []
                            for m in pca_dimension:  #pca降到m维
                                with open(cache_path, 'rb') as file:
                                    cache = pkl.load(file)
                                    embeddings = cache["embeddings"].to(device)
                                    labels = cache["labels"].to(device)
                                    norm_threshold = cache["norm_threshold"]

                                if use_pca:
                                    pca = PCA(n_components=m, svd_solver='full')
                                    # pca = PCA(n_components='mle', svd_solver='full')
                                    X = embeddings.cpu()
                                    pca.fit(X)
                                    X_pca = pca.transform(X)
                                    embeddings = torch.tensor(X_pca).to(device)
                                else:
                                    pca = None

                                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                                save_path = "gmm_params.pth"
                                torch.save({'loc': gaussians_model.loc, 'covariance_matrix': gaussians_model.covariance_matrix}, save_path)
                                storage2 = os.path.getsize(save_path)
                                os.remove(save_path)

                                start = time.time()
                                logits2, labels2, preds2 = gmm_evaluate(
                                    net,
                                    gaussians_model,
                                    pca,
                                    test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    storage_device=device,
                                )

                                ood_logits2, ood_labels2, _ = gmm_evaluate(
                                    net,
                                    gaussians_model,
                                    pca,
                                    ood_test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    storage_device=device,
                                )

                                interval2 = time.time() - start
                                _, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device, conf=True)
                                print(f"m1_auroc:{m1_auroc},m1_auprc:{m1_auprc};D={m},m2_auroc:{m2_auroc},m2_auprc:{m2_auprc}")
                                res.append([m2_auroc, m2_auprc, interval2, m])
                                pca_interval.append(interval2)
                                pca_storage.append(storage2)
                                pca_auroc.append(m2_auroc)
                                pca_auprc.append(m2_auprc)

                            m2_auroc, m2_auprc, t_ece, epsilon = [max(col) for col in zip(*res)]  #这里epsilon实际是m,t_ece是时间
                        elif args.perturbation == "pcakde":  #pca降维
                            res = []
                            use_pca = True
                            for m in [8, 16, 32, 64, 128, 256, 512]:
                                with open(cache_path, 'rb') as file:
                                    cache = pkl.load(file)
                                    embeddings = cache["embeddings"].to(device)
                                    labels = cache["labels"].to(device)
                                    norm_threshold = cache["norm_threshold"]

                                if use_pca:
                                    pca = PCA(n_components=m, svd_solver='full')
                                    # pca = PCA(n_components='mle', svd_solver='full')
                                    X = embeddings.cpu()
                                    pca.fit(X)
                                    X_pca = pca.transform(X)
                                    embeddings = torch.tensor(X_pca).to(device)
                                else:
                                    pca = None

                                kde_model = kde_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                                logits2, labels = kde_evaluate(
                                    net,
                                    kde_model,
                                    pca,
                                    test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    storage_device=device,
                                )

                                ood_logits2, ood_labels = kde_evaluate(
                                    net,
                                    kde_model,
                                    pca,
                                    ood_test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    storage_device=device,
                                )

                                _, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, maxval, device, conf=True)
                                print(f"KDE,m1_auroc:{m1_auroc},m1_auprc:{m1_auprc};D={m},m2_auroc:{m2_auroc},m2_auprc:{m2_auprc}")
                                res.append([m2_auroc, m2_auprc, -t_ece, m])
                            m2_auroc, m2_auprc, t_ece, epsilon = [max(col) for col in zip(*res)]
                        else:
                            raise ValueError("perturbation is invalid...")

                        m2_res.append([m2_auroc, m2_auprc, -t_ece, epsilon])

                if args.perturbation == "pca":
                    pca_res[saved_model_name] = {}
                    pca_res[saved_model_name]['ori_interval'] = interval1
                    pca_res[saved_model_name]['ori_storage'] = storage
                    pca_res[saved_model_name]['ori_auroc'] = m1_auroc
                    pca_res[saved_model_name]['ori_auprc'] = m1_auprc
                    pca_res[saved_model_name]['dimension'] = pca_dimension
                    pca_res[saved_model_name]['interval'] = pca_interval
                    pca_res[saved_model_name]['storage'] = pca_storage
                    pca_res[saved_model_name]['auroc'] = pca_auroc
                    pca_res[saved_model_name]['auprc'] = pca_auprc
                m2_auroc, m2_auprc, t_ece, epsilon = sorted(m2_res)[-1]  #从小到大排序，并且取最大的
                m2_auroc, m2_auprc, t_ece, _ = [max(col) for col in zip(*m2_res)]
                print(f"最优:m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f};epsilon:{epsilon},m2_auroc:{m2_auroc:.4f},m2_auprc:{m2_auprc:.4f}")
            elif (args.evaltype == "kde"):
                # Evaluate a kde model
                cache_path = saved_model_name.replace(".model", ".cache")
                load_cache = True
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

                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                logits, _, preds = gmm_evaluate(
                    net,
                    gaussians_model,
                    None,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                ood_logits, _, _ = gmm_evaluate(
                    net,
                    gaussians_model,
                    None,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )
                m1_fpr95, m1_auroc, m1_auprc = get_roc_auc_logits(logits, ood_logits, maxval, device, conf=True)
                print(f"m1_auroc:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f}")

                kde_model = kde_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                logits2, labels = kde_evaluate(
                    net,
                    kde_model,
                    None,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                ood_logits2, ood_labels = kde_evaluate(
                    net,
                    kde_model,
                    None,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                m2_fpr95, m2_auroc, m2_auprc = get_roc_auc_logits(logits2, ood_logits2, logsumexp, device, conf=True)
                epsilon = 0
                print(f"accu:{accuracy:.4f},m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f},m2_auroc:{m2_auroc:.4f},m2_auprc:{m2_auprc:.4f}")
            elif (args.evaltype == "softmax"):
                logits, _ = maxp_evaluate(
                    net,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )
                ood_logits, _ = maxp_evaluate(
                    net,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )
                _, m1_auroc, m1_auprc = get_roc_auc_logits(logits, ood_logits, confidence, device, conf=True)
                print(f"m1_auroc:{m1_auroc:.4f},m1_aupr:{m1_auprc:.4f}")

                m2_auroc = m1_auroc
                m2_auprc = m1_auprc
                epsilon = 0
                print(f"accu:{accuracy:.4f},m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f},m2_auroc:{m2_auroc:.4f},m2_auprc:{m2_auprc:.4f}")

        epsilons.append(epsilon)
        accuracies.append(accuracy)
        eces.append(ece)
        t_eces.append(abs(t_ece))

        m1_aurocs.append(m1_auroc)
        m1_auprcs.append(m1_auprc)
        m2_aurocs.append(m2_auroc)
        m2_auprcs.append(m2_auprc)

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
    res_dict["annotations"] = "m1:without noise,m2:add noise"

    saved_name = "res_" + model_save_name(
        args.model, args.sn, args.mod, args.coeff, args.seed,
        args.contrastive) + "_" + args.evaltype + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.perturbation + ".json"
    saved_dir = f"./results/run{args.run}/"
    if (not os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    with open(os.path.join(saved_dir, saved_name), "w") as f:
        json.dump(res_dict, f)
        print(f"save to {os.path.join(saved_dir,saved_name)}")

    if args.perturbation == 'pca':
        with open(os.path.join(saved_dir, saved_name.replace("pca", "pca_dimension")), "w") as f:
            json.dump(pca_res, f)
