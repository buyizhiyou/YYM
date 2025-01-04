import sys

sys.path.append("../")

import warnings
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import data_utils.ood_detection.cifar10 as cifar10
from net.vgg import vgg16
from utils.attack_utils import (bim_attack, cw_attack, deepfool_attack,
                                fgsm_attack, pgd_attack)
from utils.eval_utils import accuracy
from utils.plots_utils import save_adv

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    # 计算协方差矩阵
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net,
    loader,
    num_dim,
    dtype,
    device,
    storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.zeros((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    norms = torch.zeros(num_samples, dtype=torch.float, device=storage_device)
    loss_func = nn.CrossEntropyLoss()

    start = 0
    print("get embeddings from dataloader...")
    with torch.no_grad():
        for images, label in tqdm(loader, dynamic_ncols=True):  # 多个少batch
            images = images.to(device)
            label = label.to(device)

            # images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
            logits = net(images)
            acc = accuracy(logits, label)[0].item()
            _, pred = torch.max(logits, 1)

            # loss = loss_func(logits, pred)  #这个loss效果好一些
            # net.zero_grad()
            # loss.backward()
            # gradient = images.grad.data
            # norm_batch = torch.norm(gradient, p=1)
            out = net.feature

            end = start + len(images)
            # norms[start:end].copy_(norm_batch, non_blocking=True)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    # norm_threshold = (norms.quantile(1).item())
    norm_threshold = 0
    # print(f"norm threshold:{norm_threshold}")

    return embeddings, labels, norm_threshold


def gmm_forward(net, gaussians_model, data_B_X):
    _ = net(data_B_X)
    features_B_Z = net.feature
    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型
    # 对数概率密度，作为logits

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model,pca, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    preds_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    with torch.no_grad():
        start = 0
        total = 0
        for data, label in tqdm(loader, dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)

            logits = net(data)
            _, pred = torch.max(logits, 1)
            features_B_Z = net.feature

            if pca :
                X = features_B_Z.cpu().numpy()
                X_pca = pca.transform(X)
                features_B_Z = torch.tensor(X_pca).to(device)

            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型
    
            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            preds_N[start:end].copy_(pred, non_blocking=True)
            start = end

    return logits_N_C, labels_N, preds_N


def show_images_denorm(images_denorm, perturbed_images):
    x = np.transpose(images_denorm.cpu().detach().numpy()[0, :, :, :], (1, 2, 0))  # C X H X W  ==>   H X W X C
    x_adv = np.transpose(perturbed_images.cpu().detach().numpy()[0, :, :, :], (1, 2, 0))  # C X H X W  ==>   H X W X C
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].imshow(x)
    ax[0].set_title("Clean Example", fontsize=10)
    ax[1].imshow(x_adv)
    ax[1].set_title("Adversarial Example", fontsize=10)
    plt.savefig("test.jpg")


def gmm_evaluate_with_perturbation(
    net,
    gaussians_model,
    loader,
    device,
    num_classes,
    storage_device,
    norm_threshold=0.1,
    perturbation="fgsm",
    epsilon=0.001,
    temperature=1.0,
    inf=-1.0e+20,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    outs_N = torch.zeros((num_samples,10), dtype=torch.float, device=storage_device)
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    start = 0
    accs = []
    accs_pertubration = []
    total = 0

    # if perturbation == "fgsm":
    #     perturb = fgsm_attack
    # elif perturbation == "bim":
    #     perturb = bim_attack
    # elif perturbation == "cw":
    #     perturb = cw_attack
    # elif perturbation == "pgd":
    #     perturb = pgd_attack

    fc_grad = []

    def bp_hook(module, grad_input, grad_output):
        # 获取中间层梯度信息
        b_grad = grad_input[0]
        w_grad = grad_input[2]
        input_grad = grad_input[1]
        fc_grad.append(torch.norm(input_grad, p=1))  #1-norm
        # fc_grad.append(w_grad)

    # handler = net.fc.register_backward_hook(bp_hook)
    # handler.remove()
    #/2024_04_23_02_33_15/vgg16_sn_3.0_mod_seed_1_best.model svhn:m1_auroc1:0.9334,m1_auprc:0.9503
    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        logits = net(images)
        acc = accuracy(logits, label)[0].item()
        accs.append(acc)
        _, pred = torch.max(logits, 1)

        # 1.加扰动
        # loss = loss_func(logits / temperature, pred)
        # net.zero_grad()
        # loss.backward()
        # gradient = images.grad.data
        # norms = torch.norm(gradient, p=1).item()
        # coeff = torch.exp(torch.tensor(max(gamma * (norms - norm_threshold), 0))) - 1  #gamma=30/10/15
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2
        # perturbed_images_normalized = torch.add(images, gradient, alpha=-coeff)

        # 2. fgsm
        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])

        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get max log-probability
        loss = max_log_probs.sum()  #这个loss效果好一些
        # loss = loss_func(logits / temperature, pred)
        net.zero_grad()
        loss.backward()
        gradient = images.grad.data
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2#TODO:验证和下面哪一种效果好
        gradient = gradient.sign()
        #TODO:验证要不要/std

        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        perturbed_images_normalized = torch.add(images, gradient, alpha=epsilon)
        # perturbed_images_normalized = torch.subtract(images, gradient, alpha=epsilon)

        out = net(perturbed_images_normalized)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        features_B_Z = net.feature

        if torch.nonzero(torch.isnan(features_B_Z) == True).sum() > 0:  ##如果含有nan
            logit_B_C = torch.ones_like(logits) * float("-inf")
        else:
            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        outs_N[start:end].copy_(out.cpu().detach(), non_blocking=True) #这里使用out代替preds，因为out可以计算推出preds
        start = end

    return logits_N_C.to(device), labels_N.to(device), outs_N.to(device), sum(accs) / len(accs), sum(accs_pertubration) / len(accs_pertubration)


def gmm_evaluate_with_perturbation3(
    net,
    gaussians_model,
    loader,
    device,
    num_classes,
    storage_device,
    norm_threshold=0.1,
    perturbation="fgsm",
    epsilon=0.001,
    temperature=1.0,
    inf=-1.0e+20,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    '''
    改成-梯度
    '''
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    outs_N = torch.zeros((num_samples,10), dtype=torch.float, device=storage_device)
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    start = 0
    accs = []
    accs_pertubration = []
    total = 0

    # if perturbation == "fgsm":
    #     perturb = fgsm_attack
    # elif perturbation == "bim":
    #     perturb = bim_attack
    # elif perturbation == "cw":
    #     perturb = cw_attack
    # elif perturbation == "pgd":
    #     perturb = pgd_attack

    fc_grad = []

    def bp_hook(module, grad_input, grad_output):
        # 获取中间层梯度信息
        b_grad = grad_input[0]
        w_grad = grad_input[2]
        input_grad = grad_input[1]
        fc_grad.append(torch.norm(input_grad, p=1))  #1-norm
        # fc_grad.append(w_grad)

    # handler = net.fc.register_backward_hook(bp_hook)
    # handler.remove()
    #/2024_04_23_02_33_15/vgg16_sn_3.0_mod_seed_1_best.model svhn:m1_auroc1:0.9334,m1_auprc:0.9503
    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        logits = net(images)
        acc = accuracy(logits, label)[0].item()
        accs.append(acc)
        _, pred = torch.max(logits, 1)

        # 1.加扰动
        # loss = loss_func(logits / temperature, pred)
        # net.zero_grad()
        # loss.backward()
        # gradient = images.grad.data
        # norms = torch.norm(gradient, p=1).item()
        # coeff = torch.exp(torch.tensor(max(gamma * (norms - norm_threshold), 0))) - 1  #gamma=30/10/15
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2
        # perturbed_images_normalized = torch.add(images, gradient, alpha=-coeff)

        # 2. fgsm
        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])

        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get max log-probability
        loss = max_log_probs.sum()  #这个loss效果好一些
        # loss = loss_func(logits / temperature, pred)
        net.zero_grad()
        loss.backward()
        gradient = images.grad.data
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2#TODO:验证和下面哪一种效果好
        gradient = gradient.sign()
        #TODO:验证要不要/std

        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        # perturbed_images_normalized = torch.add(images, gradient, alpha=epsilon)
        perturbed_images_normalized = torch.subtract(images, gradient, alpha=epsilon)

        out = net(perturbed_images_normalized)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        features_B_Z = net.feature

        if torch.nonzero(torch.isnan(features_B_Z) == True).sum() > 0:  ##如果含有nan
            logit_B_C = torch.ones_like(logits) * float("-inf")
        else:
            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        outs_N[start:end].copy_(out.cpu().detach(), non_blocking=True) #这里使用out代替preds，因为out可以计算推出preds
        start = end

    return logits_N_C.to(device), labels_N.to(device), outs_N.to(device), sum(accs) / len(accs), sum(accs_pertubration) / len(accs_pertubration)


def gmm_evaluate_with_perturbation2(
    net,
    gaussians_model,
    loader,
    device,
    num_classes,
    storage_device,
    norm_threshold=0.1,
    perturbation="fgsm",
    epsilon=0.001,
    temperature=1.0,
    inf=-1.0e+20,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    '''
    输入扰动加入两次
    '''
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    outs_N = torch.zeros((num_samples,10), dtype=torch.float, device=storage_device)
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    start = 0
    accs = []
    accs_pertubration = []
    total = 0

    # if perturbation == "fgsm":
    #     perturb = fgsm_attack
    # elif perturbation == "bim":
    #     perturb = bim_attack
    # elif perturbation == "cw":
    #     perturb = cw_attack
    # elif perturbation == "pgd":
    #     perturb = pgd_attack

    fc_grad = []

    def bp_hook(module, grad_input, grad_output):
        # 获取中间层梯度信息
        b_grad = grad_input[0]
        w_grad = grad_input[2]
        input_grad = grad_input[1]
        fc_grad.append(torch.norm(input_grad, p=1))  #1-norm
        # fc_grad.append(w_grad)

    # handler = net.fc.register_backward_hook(bp_hook)
    # handler.remove()
    #/2024_04_23_02_33_15/vgg16_sn_3.0_mod_seed_1_best.model svhn:m1_auroc1:0.9334,m1_auprc:0.9503
    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        
        #第一次加入输入扰动
        logits = net(images)
        acc = accuracy(logits, label)[0].item()
        accs.append(acc)

        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])

        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get max log-probability
        loss = max_log_probs.sum()  #这个loss效果好一些

        net.zero_grad()
        loss.backward()
        gradient = images.grad.data
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2#TODO:验证和下面哪一种效果好
        gradient = gradient.sign()
        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        perturbed_images_normalized = torch.add(images, gradient, alpha=epsilon)

        #第二次加入输入扰动
        out = net(perturbed_images_normalized)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        
        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])

        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get max log-probability
        loss = max_log_probs.sum()  #这个loss效果好一些

        net.zero_grad()
        loss.backward()
        gradient = images.grad.data
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2#TODO:验证和下面哪一种效果好
        gradient = gradient.sign()
        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        perturbed_images_normalized2 = torch.add(perturbed_images_normalized, gradient, alpha=epsilon)

        out = net(perturbed_images_normalized2)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        
        features_B_Z = net.feature

        if torch.nonzero(torch.isnan(features_B_Z) == True).sum() > 0:  ##如果含有nan
            logit_B_C = torch.ones_like(logits) * float("-inf")
        else:
            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        outs_N[start:end].copy_(out.cpu().detach(), non_blocking=True) #这里使用out代替preds，因为out可以计算推出preds
        start = end

    return logits_N_C.to(device), labels_N.to(device), outs_N.to(device), sum(accs) / len(accs), sum(accs_pertubration) / len(accs_pertubration)



def gmm_evaluate_for_adv(net, gaussians_model, loader, device, num_classes, storage_device, perturbation="fgsm"):

    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    preds_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)

    start = 0
    total = 0
    #下面的攻击方式全部使用默认的参数，不同的攻击强度也影响实验结果
    if perturbation == "fgsm":
        perturb = partial(fgsm_attack)
    elif perturbation == "bim":
        perturb = partial(bim_attack)
    elif perturbation == "cw":
        perturb = partial(cw_attack)
    elif perturbation == "pgd":
        perturb = partial(pgd_attack)
    else:
        raise ValueError("perturbation is not valid")

    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        logits = net(images)
        init_prob = torch.softmax(logits, axis=1)
        clean_prob, init_pred = init_prob.max(1, keepdim=True)

        _, pred = torch.max(logits, 1)
        images_adv = perturb(net, images, label, device)
        images_adv.requires_grad = True  #images.required_grad区分,用required_grad梯度为None

        logits = net(images_adv)
        final_prob = torch.softmax(logits, axis=1)
        adv_prob, final_pred = final_prob.max(1, keepdim=True)

        _, pred = torch.max(logits, 1)
        features_B_Z = net.feature
        logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型

        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        preds_N[start:end].copy_(pred.cpu().detach(), non_blocking=True)
        start = end
    
    #可视化中间攻击的图片
    # images = images[0].cpu().detach().numpy()
    # images_adv = images_adv[0].cpu().detach().numpy()
    # init_pred = init_pred[0].item()
    # final_pred = final_pred[0].item()
    # clean_prob = clean_prob[0].item()
    # adv_prob = adv_prob[0].item()
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2023, 0.1994, 0.2010]
    # save_adv("./results/images/adv.png", images, images_adv, init_pred, final_pred, clean_prob, adv_prob, mean, std)


    return logits_N_C.to(device), labels_N.to(device), preds_N.to(device)


def gmm_evaluate_with_perturbation_for_adv(
    net,
    gaussians_model,
    loader,
    device,
    num_classes,
    storage_device,
    norm_threshold=0.1,
    perturbation="fgsm",
    epsilon=0.001,
    temperature=1.0,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    preds_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    start = 0
    accs = []
    accs_pertubration = []
    total = 0

    #下面的攻击方式全部使用默认的参数，不同的攻击强度也影响实验结果
    if perturbation == "fgsm":
        perturb = partial(fgsm_attack)
    elif perturbation == "bim":
        perturb = partial(bim_attack)
    elif perturbation == "cw":
        perturb = partial(cw_attack)
    elif perturbation == "pgd":
        perturb = partial(pgd_attack)
    else:
        raise ValueError("perturbation is not valid")

    #/2024_04_23_02_33_15/vgg16_sn_3.0_mod_seed_1_best.model svhn:m1_auroc1:0.9334,m1_auprc:0.9503
    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        logits = net(images)
        acc = accuracy(logits, label)[0].item()
        accs.append(acc)
        _, pred = torch.max(logits, 1)

        # images_adv = perturb(net, images, pred, device)
        images_adv = perturb(net, images, label, device)
        images_adv.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        logits = net(images_adv)
        _, pred = torch.max(logits, 1)

        # embedding = net.feature
        # log_probs = gaussians_model.log_prob(embedding[:, None, :])
        # max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
        # loss = -max_log_probs.sum()
        # loss = loss_func(logits / temperature, pred)  #这个loss效果好一些
        # net.zero_grad()
        # loss.backward()

        embedding = net.feature
        log_probs = gaussians_model.log_prob(embedding[:, None, :])
        max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
        loss = max_log_probs.sum()  #这个loss效果好一些
        # loss = loss_func(logits / temperature, pred)
        net.zero_grad()
        loss.backward()
        gradient = images_adv.grad.data
        # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2#TODO:验证和下面哪一种效果好
        gradient = gradient.sign()
        #TODO:验证要不要/std
        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        perturbed_images_normalized = torch.add(images_adv, gradient, alpha=epsilon)

        # # #2.加扰动
        # gradient = images_adv.grad.data
        # norms = torch.norm(gradient).item()
        # coeff = epsilon * torch.exp(torch.tensor(max(30 * (norms - 0.1), 0)))
        # # gradient = (gradient - gradient.min() / (gradient.max() - gradient.min()) - 0.5) * 2
        # # gradient = gradient.sign()
        # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / std[0])
        # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / std[1])
        # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / std[2])
        # perturbed_images_normalized = torch.add(images_adv, gradient, alpha=coeff)

        out = net(perturbed_images_normalized)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        features_B_Z = net.feature
        if torch.nonzero(torch.isnan(features_B_Z) == True).sum() > 0:  ##如果含有nan
            logit_B_C = torch.ones_like(logits) * float("-inf")
        else:
            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        preds_N[start:end].copy_(pred.cpu().detach(), non_blocking=True)
        start = end

    # return logits_N_C.to(device), labels_N.to(device), preds_N.to(device)
    return logits_N_C.to(device), labels_N.to(device), preds_N.to(device), sum(accs) / len(accs), sum(accs_pertubration) / len(accs_pertubration)


def gradient_norm_collect(net, gaussians_model, loader, device, storage_device, temperature=1, norm=1,type=1):
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros(num_samples, dtype=torch.float, device=storage_device)

    loss_func = nn.CrossEntropyLoss()
    start = 0

    for images, label in tqdm(loader, dynamic_ncols=True):
        images = images.to(device)
        label = label.to(device)
        label = label % 10
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        out = net(images)
        _, pred = torch.max(out, 1)

      
        if type==1:  #1. 第一种形式loss
            embedding = net.feature
            log_probs = gaussians_model.log_prob(embedding[:, None, :])
            max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
            loss = max_log_probs.sum()
        else:
            # #2. 第二种形式loss
            loss = -loss_func(out / temperature, pred)

        net.zero_grad()
        loss.backward()

        gradient = images.grad.data
        gradient_norms = -1 * torch.norm(gradient, p=norm, dim=(1, 2, 3))
        # gradient_norms = torch.linalg.matrix_norm(gradient,ord=norm,dim=(2,3)).sum(dim=1)
        end = start + len(images)
        logits_N_C[start:end].copy_(gradient_norms.cpu().detach(), non_blocking=True)
        start = end

    return logits_N_C.to(device)


def maxp_evaluate_with_perturbation(
    net,
    loader,
    device,
    num_classes,
    storage_device,
    epsilon=0.001,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)

    loss_func = nn.CrossEntropyLoss()
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    start = 0
    for data, label in tqdm(loader, dynamic_ncols=True):
        data = data.to(device)
        data.requires_grad = True  #data.required_grad区分,用required_grad梯度为None

        label = label.to(device)
        label = label % 10

        out = net(data)
        _, pred = torch.max(out, 1)

        perturbed_data = fgsm_attack(net, data, pred, device)

        perturbed_data_normalized = (perturbed_data - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        logit_B_C = net(perturbed_data_normalized)  # 每个batch计算logits,再合并

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(data)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        start = end

    return logits_N_C.to(device), labels_N.to(device)


def maxp_evaluate(net, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.zeros((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.zeros(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader, dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = net(data)  # 每个batch计算logits,再合并

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):
    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):
    '''
    在这段代码中，每个类别构建了一个高斯分布，共num_classes个。例如，如果 num_classes=10，则 gmm 包含 10 个独立的多元正态分布。
    '''
    with torch.no_grad():
        # 对每个类别，求均值，协方差
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack([torch.cov(embeddings[labels == c].T) for c in range(num_classes)])
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:  #协方差矩阵要求正定,但是样本协方差矩阵不一定正定,所以加上对角阵转化为正定的
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                    device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=(classwise_cov_features + jitter),
                )
                # MultivariateNormal(loc: torch.Size([10, 2048]), covariance_matrix: torch.Size([10, 2048, 2048]))
            except ValueError as e:
                # if "The parameter covariance_matrix has invalid values" in str(e):
                continue
            break

    return gmm, jitter_eps


# test
if __name__ == '__main__':
    device = "cuda:1"
    train_embeddings = torch.load("train_embeddings.pth").to(device)
    labels = torch.load("labels.pth").to(device)
    gaussians_model, jitter_eps = gmm_fit(embeddings=train_embeddings, labels=labels, num_classes=10)
    net = vgg16(
        spectral_normalization=True,
        mod=True,
        coeff=3.0,
        num_classes=10,
        temp=1.0,
    ).to(device)
    net.eval()
    net.drop.training = True
    #这里的参数sample_size对结果有影响
    test_loader = cifar10.get_test_loader(root="../data", batch_size=64, pin_memory=0, size=32, sample_size=100)
    logits2, labels2 = gmm_evaluate_with_perturbation(
        net,
        gaussians_model,
        test_loader,
        device=device,
        num_classes=10,
        storage_device=device,
    )

#     logits2, labels2 = maxp_evaluate_with_perturbation(
#         net,
#         test_loader,
#         device=device,
#         num_classes=10,
#         storage_device=device,
#     )
