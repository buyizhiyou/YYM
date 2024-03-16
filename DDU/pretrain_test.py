#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_test.py
@Time    :   2024/03/04 14:34:59
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
"""
Script to evaluate a single model. 
"""
import os
import json
from tqdm import tqdm
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch
import clip
import numpy as np
import gc

from torchvision import datasets
from torchvision.transforms import Resize

# Import metrics to compute
from metrics.uncertainty_confidence import entropy, logsumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.train_utils import model_save_name
from utils.args import eval_args

# Dataset params
dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "tiny_iamgenet": 200
}

if __name__ == "__main__":

    args = eval_args().parse_args()

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if cuda else "cpu")

    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    data_dir = "./data"
    # 数据增强
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomResizedCrop((224, 224),
                                                 scale=(0.9, 1),
                                                 ratio=(0.9, 1.1)),
        torchvision.transforms.ColorJitter(brightness=0.2,
                                           contrast=0.2,
                                           saturation=0.2,
                                           hue=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    train_loader = torch.utils.data_utils.DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    test_loader = torch.utils.data_utils.DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    lastlayer_dims = model.fc.in_features
    model.fc = torch.nn.Linear(lastlayer_dims, num_classes)
    model = model.to(device)

    # loss_func = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     0.1,
    #     momentum=0.9,
    #     weight_decay=5e-4,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                        T_max=200,
    #                                                        eta_min=1e-7)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    # best_acc = 0
    # for epoch in range(350):
    #     total_train_loss = 0
    #     total_test_loss = 0
    #     total_train_acc = 0
    #     total_test_acc = 0
    #     for (b_x, b_y) in tqdm(train_loader):  # 分配batch data
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #         output = model(b_x)  # 先将数据放到cnn中计算output
    #         loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
    #         total_train_loss += loss.item()

    #         optimizer.zero_grad()  # 清除之前学到的梯度的参数
    #         loss.backward()  # 反向传播，计算梯度
    #         optimizer.step()  # 应用梯度

    #         train_pred_y = torch.max(output, 1)[1]
    #         train_accuracy = (train_pred_y == b_y).sum()
    #         total_train_acc += train_accuracy.item()

    #     scheduler.step()
    #     tqdm.write("epoch:{}, loss:{:.4f},acc:{:.2f}%".format(epoch, total_train_loss / train_size,100 * total_train_acc / train_size))

    #     for (test_x, test_y) in tqdm(test_loader):
    #         test_x = test_x.to(device)
    #         test_y = test_y.to(device)
    #         test_output = model(test_x)
    #         pred_y = torch.max(test_output, 1)[1]
    #         test_loss = loss_func(test_output, test_y)
    #         total_test_loss += test_loss.item()
    #         accuracy = (pred_y == test_y).sum()
    #         total_test_acc += accuracy.item()

    #     acc = 100 * total_test_acc / test_size
    #     if(acc>best_acc):
    #         best_acc = acc
    #         torch.save(model.state_dict(),"saved_models/pretrain/resnet50_fintuning.pth")

    #     tqdm.write("epoch:{},loss:{:.4f},acc: {:.2f}%".format(epoch, total_test_loss / test_size,acc))

    model.load_state_dict(torch.load("saved_models/pretrain/resnet50_fintuning.pth"))
    model.eval()

    dataset = datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transform,
    )
    ood_test_loader = torch.utils.data_utils.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    embeddings = []
    labels = []
    torch_resize = Resize((224, 224))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # 如果你想feature的梯度能反向传播，那么去掉 detach（）
            activation[name] = input[0]
        return hook

    model.fc.register_forward_hook(get_activation('fc'))
    for batch_x, batch_y in tqdm(train_loader):
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
        model(batch_x).to(torch.float32)
        features = activation['fc']
        # features /= features.norm(dim=-1, keepdim=True)

        embeddings.append(features.cpu().detach().numpy())
        labels.append(batch_y.cpu().detach().numpy())

    embeddings = np.concatenate(embeddings, 0)
    labels = np.concatenate(labels, 0)

    embeddings = torch.from_numpy(embeddings).to(device)
    labels = torch.from_numpy(labels).to(device)
    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings,
                                          labels=labels,
                                          num_classes=num_classes)

    gc.collect()
    torch.cuda.empty_cache()

    logits = []
    total_test_acc = 0
    for batch_x, batch_y in tqdm(test_loader):
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
  
        test_output = model(batch_x).to(torch.float32).cpu().detach()
        pred_y = torch.max(test_output, 1)[1]
        accuracy = (pred_y == batch_y.cpu().detach()).sum().item()
        total_test_acc += accuracy

        features = activation['fc']
        # features /= features.norm(dim=-1, keepdim=True)
        log_probs = gaussians_model.log_prob(features[:, None, :])

        logits.append(log_probs.cpu().detach().numpy())


    acc = 100 * total_test_acc / test_size
    logits = np.concatenate(logits, 0)

    gc.collect()
    torch.cuda.empty_cache()
    
    print(torch.cuda.memory_summary())
    ood_logits = []
    for batch_x, batch_y in tqdm(ood_test_loader):
        batch_x = batch_x.to(device)
        batch_x = torch_resize(batch_x)
        batch_y = batch_y.to(device)
        model(batch_x)
        features = activation['fc']
        # features /= features.norm(dim=-1, keepdim=True)
        log_probs = gaussians_model.log_prob(features[:, None, :])

        ood_logits.append(log_probs.cpu().detach().numpy())

    ood_logits = np.concatenate(ood_logits, 0)

    logits = torch.from_numpy(logits).to(device)
    ood_logits = torch.from_numpy(ood_logits).to(device)
    (_, _, _), (_, _,
                _), m1_auroc, m1_auprc = get_roc_auc_logits(logits,
                                                            ood_logits,
                                                            logsumexp,
                                                            device,
                                                            confidence=True)
    (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(
        logits, ood_logits, entropy, device)

    t_m1_auroc = m1_auroc
    t_m1_auprc = m1_auprc
    t_m2_auroc = m2_auroc
    t_m2_auprc = m2_auprc

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = acc
    res_dict["mean"]["t_m1_auroc"] = t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = t_m2_auprc.item()

    res_dict["info"] = vars(args)

    with open(
            "./logs/res_pretrain_" + model_save_name(args.model, args.sn, args.mod,
                                                 args.coeff, args.seed, False) + "_" +
            args.model_type + "_" + args.dataset + "_" + args.ood_dataset +
            ".json",
            "w",
    ) as f:
        json.dump(res_dict, f)
