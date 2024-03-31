"""
Script to evaluate a single model. 
"""
import os
import json
import math
import torch
import glob
import argparse
import torch.backends.cudnn as cudnn

# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet

# Import network models
from net.lenet import lenet
# from net.resnet import resnet18, resnet50, resnet101, resnet110, resnet152
from net.resnet2 import resnet18, resnet50, resnet101, resnet110, resnet152
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import metrics to compute
from metrics.classification_metrics import (test_classification_net,
                                            test_classification_net_logits,
                                            test_classification_net_ensemble)
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp, confidence, sumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit, maxp_evaluate
from utils.kde_utils import kde_evaluate, kde_fit
from utils.lof_utils import lof_evaluate,ldaf_evaluate
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import eval_args

# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "tiny_iamgenet": 200
}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "tiny_imagenet": tiny_imagenet
}

# Mapping model name to model function
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

model_to_num_dim = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "wide_resnet": 640,
    "vgg16": 512
}

torch.backends.cudnn.enabled = False

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
    test_loader = dataset_loader[args.dataset].get_test_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        pin_memory=args.gpu)
    ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        pin_memory=args.gpu)

    # Evaluating the models
    accuracies = []

    # Pre temperature scaling
    # m1 - Uncertainty/Confidence Metric 1 for deterministic model: logsumexp of probability density
    # m2 - Uncertainty/Confidence Metric 2 for deterministic model: max p
    eces = []
    m1_aurocs = []
    m1_auprcs = []
    m2_aurocs = []
    m2_auprcs = []

    topt = None
    model_name = model_load_name(args.model, args.sn, args.mod, args.coeff,
                                 args.seed, args.contrastive) + "_best.model"
    model_files = glob.glob(f"{args.load_loc}/run{args.run}/*/{model_name}")

    for saved_model_name in model_files:
        print(f"Run {args.run}, Evaluating: {saved_model_name}")
        #load dataset
        train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
            root=args.dataset_root,
            batch_size=args.batch_size,
            augment=args.data_aug,
            val_seed=(args.seed),
            val_size=0.0,
            pin_memory=args.gpu,
        )

        #load model
        print(f"load {saved_model_name}")
        net = models[args.model](
            spectral_normalization=args.sn,
            mod=args.mod,
            coeff=args.coeff,
            num_classes=num_classes,
            temp=1.0,
        )
        if args.gpu:
            net.to(device)
            cudnn.benchmark = True
        net.load_state_dict(torch.load(str(saved_model_name)), strict=False)
        net.eval()

        (
            conf_matrix,
            accuracy,
            labels_list,
            predictions,
            confidences,
        ) = test_classification_net(net, test_loader, device)
        ece = expected_calibration_error(confidences,
                                         predictions,
                                         labels_list,
                                         num_bins=15)

        if (args.model_type == "gmm"):
            # Evaluate a GMM model
            print("GMM Model")
            embeddings, labels = get_embeddings(
                net,
                train_loader,
                num_dim=model_to_num_dim[args.model],
                dtype=torch.double,
                device=device,
                storage_device=device,
            )

            test_embeddings, test_labels = get_embeddings(
                net,
                test_loader,
                num_dim=model_to_num_dim[args.model],
                dtype=torch.double,
                device=device,
                storage_device=device,
            )

            ood_test_embeddings, ood_labels = get_embeddings(
                net,
                ood_test_loader,
                num_dim=model_to_num_dim[args.model],
                dtype=torch.double,
                device=device,
                storage_device=device,
            )

            # lof_evaluate(embeddings.cpu().detach().numpy(),
            #              test_embeddings.cpu().detach().numpy(),
            #              ood_test_embeddings.cpu().detach().numpy())

            try:
                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings,
                                                      labels=labels,
                                                      num_classes=num_classes)
                
                logits, labels = gmm_evaluate(
                    net,
                    gaussians_model,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                ood_logits, ood_labels = gmm_evaluate(
                    net,
                    gaussians_model,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                # logits2, labels2 = maxp_evaluate(
                #     net,
                #     test_loader,
                #     device=device,
                #     num_classes=num_classes,
                #     storage_device=device,
                # )

                # ood_logits2, ood_labels2 = maxp_evaluate(
                #     net,
                #     ood_test_loader,
                #     device=device,
                #     num_classes=num_classes,
                #     storage_device=device,
                # )

                _, _, m1_auroc, m1_auprc = get_roc_auc_logits(logits,
                                                              ood_logits,
                                                              logsumexp,
                                                              device,
                                                              conf=True)

                m2_auroc,m2_auprc = ldaf_evaluate(gaussians_model,embeddings,test_embeddings,ood_test_embeddings)


                # _, _, m2_auroc, m2_auprc = get_roc_auc_logits(
                #     logits2, ood_logits2, confidence, device)  #最大概率

            except RuntimeError as e:
                print("Runtime Error caught: " + str(e))
                continue
        elif (args.model_type == "kde"):
            # Evaluate a kde model
            print("kde Model")
            embeddings, labels = get_embeddings(
                net,
                train_loader,
                num_dim=model_to_num_dim[args.model],
                dtype=torch.double,
                device=device,
                storage_device=device,
            )

            try:
                kde_model = kde_fit(embeddings=embeddings,
                                    labels=labels,
                                    num_classes=num_classes)
                logits, labels = kde_evaluate(
                    net,
                    kde_model,
                    test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                ood_logits, ood_labels = kde_evaluate(
                    net,
                    kde_model,
                    ood_test_loader,
                    device=device,
                    num_classes=num_classes,
                    storage_device=device,
                )

                (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(
                    logits, ood_logits, logsumexp, device, confidence=True)
                (_, _,
                 _), (_, _,
                      _), m2_auroc, m2_auprc = get_roc_auc_logits(logits,
                                                                  ood_logits,
                                                                  entropy,
                                                                  device,
                                                                  conf=True)

            except RuntimeError as e:
                print("Runtime Error caught: " + str(e))
                continue

        accuracies.append(accuracy)
        # Pre-temperature results
        eces.append(round(ece, 4))
        m1_aurocs.append(round(m1_auroc, 4))
        m1_auprcs.append(round(m1_auprc, 4))
        m2_aurocs.append(round(m2_auroc, 4))
        m2_auprcs.append(round(m2_auprc, 4))

        print(
            f"{saved_model_name} accu:{accuracy:.4f},ece:{ece:.6f},m1_auroc1:{m1_auroc:.4f},m1_auprc:{m1_auprc:.4f},m2_auroc:{m2_auroc:.4f},m2_auprc:{m2_auprc:.4f}"
        )

    accuracy_tensor = torch.tensor(accuracies)
    ece_tensor = torch.tensor(eces)
    m1_auroc_tensor = torch.tensor(m1_aurocs)
    m1_auprc_tensor = torch.tensor(m1_auprcs)
    m2_auroc_tensor = torch.tensor(m2_aurocs)
    m2_auprc_tensor = torch.tensor(m2_auprcs)

    mean_accuracy = torch.mean(accuracy_tensor)
    mean_ece = torch.mean(ece_tensor)
    mean_m1_auroc = torch.mean(m1_auroc_tensor)
    mean_m1_auprc = torch.mean(m1_auprc_tensor)
    mean_m2_auroc = torch.mean(m2_auroc_tensor)
    mean_m2_auprc = torch.mean(m2_auprc_tensor)

    std_accuracy = torch.std(accuracy_tensor) / math.sqrt(
        accuracy_tensor.shape[0])
    std_ece = torch.std(ece_tensor) / math.sqrt(ece_tensor.shape[0])
    std_m1_auroc = torch.std(m1_auroc_tensor) / math.sqrt(
        m1_auroc_tensor.shape[0])
    std_m1_auprc = torch.std(m1_auprc_tensor) / math.sqrt(
        m1_auprc_tensor.shape[0])
    std_m2_auroc = torch.std(m2_auroc_tensor) / math.sqrt(
        m2_auroc_tensor.shape[0])
    std_m2_auprc = torch.std(m2_auprc_tensor) / math.sqrt(
        m2_auprc_tensor.shape[0])

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["m1_auroc"] = mean_m1_auroc.item()
    res_dict["mean"]["m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["m2_auprc"] = mean_m2_auprc.item()

    res_dict["std"] = {}
    res_dict["std"]["accuracy"] = std_accuracy.item()
    res_dict["std"]["ece"] = std_ece.item()
    res_dict["std"]["m1_auroc"] = std_m1_auroc.item()
    res_dict["std"]["m1_auprc"] = std_m1_auprc.item()
    res_dict["std"]["m2_auroc"] = std_m2_auroc.item()
    res_dict["std"]["m2_auprc"] = std_m2_auprc.item()

    res_dict["values"] = {}
    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["m1_auroc"] = m1_aurocs
    res_dict["values"]["m1_auprc"] = m1_auprcs
    res_dict["values"]["m2_auroc"] = m2_aurocs
    res_dict["values"]["m2_auprc"] = m2_auprcs

    res_dict["info"] = vars(args)
    res_dict["files"] = model_files


    saved_name = "res_" + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed,args.contrastive) + "_" \
                            +args.model_type + "_" + args.dataset + "_" + args.ood_dataset +".json"
    saved_dir = f"./results/run{args.run}/"
    if (not os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    with open(
            os.path.join(saved_dir, saved_name),
            "w",
    ) as f:
        json.dump(res_dict, f)
        print(f"save to {os.path.join(saved_dir,saved_name)}")
