import sys

sys.path.append("../")

from matplotlib import pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import data_utils.ood_detection.cifar10 as cifar10
from utils.eval_utils import accuracy
from utils.attack_utils import fgsm_attack, bim_attack, deepfool_attack, pgd_attack, cw_attack

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    # 计算协方差矩阵
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net,
    loader: torch.utils.data.DataLoader,
    num_dim: int,
    dtype,
    device,
    storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        print("get embeddings from dataloader...")
        for data, label in tqdm(loader):  # 多个少batch
            data = data.to(device)
            label = label.to(device)

            out = net(data)
            out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):
    _ = net(data_B_X)
    features_B_Z = net.feature
    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型
    # 对数概率密度，作为logits

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        total = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            _ = net(data)
            features_B_Z = net.feature
            logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

            # total += data.shape[0]
            # if (total > 10000):
            #     break

    return logits_N_C, labels_N


def fgsm(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def gmm_evaluate_with_perturbation(
    net,
    gaussians_model,
    loader,
    device,
    num_classes,
    storage_device,
    perturbation="fgsm",
    epsilon=0.001,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    loss_func = nn.CrossEntropyLoss()
    start = 0
    accs = []
    accs_pertubration = []
    total = 0

    if perturbation == "fgsm":
        perturb = fgsm_attack
    elif perturbation == "bim":
        perturb = bim_attack
    elif perturbation == "cw":
        perturb = cw_attack
    elif perturbation == "pgd":
        perturb = pgd_attack

    for images, label in tqdm(loader):
        images = images.to(device)
        label = label.to(device)
        label = label % 10
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        out = net(images)
        acc = accuracy(out, label)[0].item()
        accs.append(acc)
        _, pred = torch.max(out, 1)

        #1.加扰动
        perturbed_images_normalized = perturb(net, images, label, device)

        #2.加扰动
        # embedding = net.feature
        # log_probs = gaussians_model.log_prob(embedding[:, None, :])
        # max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
        # loss = max_log_probs.sum()
        # # loss = -loss_func(out, pred)  #这个loss效果好一些

        # net.zero_grad()
        # loss.backward()

        # # Normalizing the gradient to binary in {0, 1}
        # gradient = (torch.ge(images.grad.data, 0))
        # gradient = (gradient.float() - 0.5) * 2  #这两行代码等价于sign(x)  torch.Size([128, 3, 32, 32])
        # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) /std[0])
        # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) /std[1])
        # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) /std[2])

        # # Adding small perturbations to images
        # perturbed_images_normalized = torch.add(images, gradient, alpha=epsilon)

        # 3.加扰动
        # images_grad = images.grad.data
        # images_denorm = images * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        # perturbed_images = fgsm(images_denorm, epsilon, images_grad)  #x_hat = x+epsilon*gradient
        # # # x = np.transpose(images_denorm.cpu().detach().numpy()[0,:,:,:], (1, 2, 0))  # C X H X W  ==>   H X W X C
        # # # x_adv = np.transpose(perturbed_images.cpu().detach().numpy()[0,:,:,:], (1, 2, 0))  # C X H X W  ==>   H X W X C
        # # # fig, ax = plt.subplots(1, 2, figsize=(4, 2))
        # # # ax[0].imshow(x)
        # # # ax[0].set_title("Clean Example", fontsize=10)
        # # # ax[1].imshow(x_adv)
        # # # ax[1].set_title("Adversarial Example", fontsize=10)
        # # # plt.savefig("test.jpg")
        # # # Reapply normalization
        # perturbed_images_normalized = (perturbed_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        out = net(perturbed_images_normalized)
        acc = accuracy(out, label)[0].item()
        accs_pertubration.append(acc)
        features_B_Z = net.feature
        logit_B_C = gaussians_model.log_prob(features_B_Z[:, None, :])  # torch.Size([128, 10]),每个类别一个多元高斯模型

        logit_B_C = logit_B_C.cpu().detach()
        end = start + len(images)
        logits_N_C[start:end].copy_(logit_B_C.cpu().detach(), non_blocking=True)
        labels_N[start:end].copy_(label.cpu().detach(), non_blocking=True)
        start = end

        total += images.shape[0]
        if (total > 10000):
            break

    return logits_N_C.to(device), labels_N.to(device), sum(accs) / len(accs), sum(accs_pertubration) / len(accs_pertubration)


def gradient_norm_collect(net, gaussians_model, loader, device, storage_device, norm=1):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty(num_samples, dtype=torch.float, device=storage_device)

    loss_func = nn.CrossEntropyLoss()
    start = 0

    for images, label in tqdm(loader):
        images = images.to(device)
        label = label.to(device)
        label = label % 10
        images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
        out = net(images)
        _, pred = torch.max(out, 1)

        #1. 第一种形式loss
        # embedding = net.feature
        # log_probs = gaussians_model.log_prob(embedding[:, None, :])
        # max_log_probs = log_probs.max(1, keepdim=True)[0]  # get the index of the max log-probability
        # loss = max_log_probs.sum()

        # #2. 第二种形式loss
        loss = -loss_func(out, pred)

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
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    loss_func = nn.CrossEntropyLoss()
    std = torch.tensor(std).to(device)
    mean = torch.tensor(mean).to(device)
    start = 0
    for data, label in tqdm(loader):
        data = data.to(device)
        data.requires_grad = True  #data.required_grad区分,用required_grad梯度为None

        label = label.to(device)
        label = label % 10

        out = net(data)
        _, pred = torch.max(out, 1)
        loss = -loss_func(out, pred)

        net.zero_grad()
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data
        data_denorm = data * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        perturbed_data = fgsm(data_denorm, epsilon, data_grad)

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
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
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
# if __name__ == '__main__':
#     device = "cuda:0"
#     train_embeddings = torch.load("train_embeddings.pth")
#     test_embeddings = torch.load("test_embeddings.pth")
#     ood_test_embeddings = torch.load("ood_test_embeddings.pth")
#     labels = torch.load("labels.pth")
#     gaussians_model, jitter_eps = gmm_fit(embeddings=train_embeddings, labels=labels, num_classes=10)
#     net = resnet50(
#         spectral_normalization=True,
#         mod=True,
#         coeff=3.0,
#         num_classes=10,
#         temp=1.0,
#     ).to(device)
#     net.eval()
#     net.drop.training = True
#     test_loader = cifar10.get_test_loader(root="../data", batch_size=64, pin_memory=0)
#     # logits2, labels2 = gmm_evaluate_with_perturbation(
#     #     net,
#     #     gaussians_model,
#     #     test_loader,
#     #     device=device,
#     #     num_classes=10,
#     #     storage_device=device,
#     # )

#     logits2, labels2 = maxp_evaluate_with_perturbation(
#         net,
#         test_loader,
#         device=device,
#         num_classes=10,
#         storage_device=device,
#     )
