# %% [markdown]
# ### Load models
# 

# %%
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import rc
import re
import pickle as pkl 

sys.path.append("../")
# Import dataloaders
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet

# Network architectures
from net.lenet import lenet
from net.resnet import resnet50
from net.vgg import vgg16
from net.wide_resnet import wrn
from net.vit import vit

from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit, maxp_evaluate, gmm_evaluate_with_perturbation, maxp_evaluate_with_perturbation
from metrics.uncertainty_confidence import entropy, logsumexp, confidence, sumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits

device = torch.device("cuda:1")
# Dataset params
dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "lsun": 10, "tiny_iamgenet": 200}
dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn, "mnist": mnist, "lsun": lsun, "tiny_imagenet": tiny_imagenet}

# Mapping model name to model function
models = {
    "lenet": lenet,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
    "vit":vit
}

model_to_num_dim = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048, "wide_resnet": 640, "vgg16": 512,"vit":768}

batch_size = 256
dataset = "cifar10"
ood_dataset = "svhn"
dataset_root = "../data"
# model = "vgg16"
# saved_model_name = "../saved_models/run17/vgg16_sn_3.0_mod_seed_1/2024_05_27_17_51_26/vgg16_sn_3.0_mod_seed_1_best.model"

model = "vit"
saved_model_name = "../saved_models/run22/vit_sn_3.0_mod_seed_1/2024_09_21_16_49_54/vit_sn_3.0_mod_seed_1_best.model"

# Taking input for the dataset
num_classes = dataset_num_classes[dataset]
test_loader = dataset_loader[dataset].get_test_loader(root=dataset_root, batch_size=batch_size)
ood_test_loader = dataset_loader[ood_dataset].get_test_loader(root=dataset_root, batch_size=batch_size)

#load model
print(f"load {saved_model_name}")
net = models[model](
    spectral_normalization=True,
    mod=True,
    num_classes=num_classes,
    temp=1.0,
).to(device)

net.load_state_dict(torch.load(str(saved_model_name), map_location=device), strict=True)
net.eval()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sns.set_style('whitegrid')
sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

rc('text', usetex=False)


def plot_activation(data1, data2, title1, title2):
    # 使用matplotlib绘制特征图
    fig, axes = plt.subplots(2, 1)
    plt.tight_layout()
    minv = np.min(data1)
    maxv = np.max(data1)

    im1 = axes[0].imshow(data1, cmap='viridis')  # cmap参数指定色彩映射
    axes[0].set_title(title1)
    im1.set_clim(minv, maxv)
    im2 = axes[1].imshow(data2, cmap='viridis')  # cmap参数指定色彩映射
    im2.set_clim(minv, maxv)
    axes[1].set_title(title2)

    fig.colorbar(im1, ax=axes)


if not os.path.exists(f"pngs/{model}/"):
    os.makedirs(f"pngs/{model}/")

train_loader, val_loader = dataset_loader[dataset].get_train_valid_loader(
    root=dataset_root,
    batch_size=512,
    augment=True,
    val_seed=1,
    val_size=0.1,
)

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
        num_dim=model_to_num_dim["vit"],
        dtype=torch.double,
        device=device,
        storage_device=device,
    )
    cache = {"embeddings": embeddings.cpu(), "labels": labels.cpu(), "norm_threshold": norm_threshold}
    with open(cache_path, "wb") as f:
        pkl.dump(cache, f)


gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)

def calculate_gradients_activation(
    net,
    gaussians_model,
    layer_name,
    loader,
    device,
    wrt="input",
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
    norm=1,
):
    layer_grad = None
    def bp_hook(module, grad_input, grad_output):
        # 获取中间层梯度信息
        nonlocal layer_grad
        input_grad = grad_input[0]
        # w_grad = grad_input[1]
        # input_grad = grad_input[2]#TODO: 当使用sn时，conv层的bias都为false，这时候这样的,当bias为true时候，需要再次验证???
        if wrt == "input":
            layer_grad = input_grad  #1-norm
        # elif wrt == "weight":
        #     layer_grad = w_grad
        # else:
        #     raise ValueError(f"{wrt} is invalid...")

    grads = []

    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    if layer_name in ["image"]:
        print(f"register backward hook for {layer_name}")
        handler = net.conv_proj.register_backward_hook(bp_hook)
    elif layer_name in range(12):
        print(f"register backward hook for {layer_name}")
        handler = net.encoder.layers[layer_name].mlp[0].register_backward_hook(bp_hook)
    else:
        raise ValueError(f"{layer_name} is invalid...")
    # import pdb;pdb.set_trace()
    cnt = 0
    for data, label in tqdm(loader, dynamic_ncols=True):
        data = data.to(device)
        label = label.to(device)

        data.requires_grad = True  #data.required_grad区分,用required_grad梯度为None
        out = net(data)

        # #1. 第一种形式loss,log_density
        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])
        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
        loss = max_log_probs.sum()

        # # #2. 第二种形式loss,crossEntropy
        # loss = loss_func(out, pred)

        net.zero_grad()
        loss.backward()

        grads.append(layer_grad.cpu().detach())
        

        cnt += data.shape[0]
        if cnt > 1000:
            break

    if wrt=="input":
        import pdb;pdb.set_trace()
        grads = torch.concat(grads,dim=0)
        activation = torch.mean(grads,dim=(0,1))
    else:
        grads =  torch.mean(torch.stack(grads),dim=0)
        activation = grads.reshape((-1, max(grads.shape)))

    return activation


layer = 1
wrt="input"
test_loader = dataset_loader[dataset].get_test_loader(root=dataset_root, batch_size=1)
ood_test_loader = dataset_loader[ood_dataset].get_test_loader(root=dataset_root, batch_size=1)
grads = calculate_gradients_activation(
    net,
    gaussians_model,
    layer,
    test_loader,
    device=device,
    norm=1,
)
ood_grads = calculate_gradients_activation(
    net,
    gaussians_model,
    layer,
    ood_test_loader,
    device=device,
    norm=1,
)
import pdb;pdb.set_trace()
plot_activation(np.abs(grads.numpy()), np.abs(ood_grads.numpy()), f"ind_test:{layer}", f"ood_test:{layer}")
plt.savefig(f'pngs/{model}/{layer}_grad_wrt_input_activation.png', bbox_inches='tight')




