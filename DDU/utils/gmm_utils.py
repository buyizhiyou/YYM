import torch
from torch import nn
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
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
    embeddings = torch.empty((num_samples, num_dim),
                             dtype=dtype,
                             device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        print("get embeddings from dataloader...")
        for data, label in tqdm(loader):  #多个少batch
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        _ = net.module(data_B_X)  #torch.Size([128, 10]) 这一个不用
        features_B_Z = net.module.feature  #torch.Size([128, 2048]) 用这一个，embedding
    else:
        _ = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(
        features_B_Z[:, None, :])  #torch.Size([128, 10]),每个类别一个多元高斯模型
    #对数概率密度，作为logits

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes,
                 storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes),
                             dtype=torch.float,
                             device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)
            logit_B_C = gmm_forward(net, gaussians_model,
                                    data)  #每个batch计算logits,再合并

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
        #对每个类别，求均值，协方差
        classwise_mean_features = torch.stack([
            torch.mean(embeddings[labels == c], dim=0)
            for c in range(num_classes)
        ])
        # classwise_cov_features = torch.stack([
        #     centered_cov_torch(embeddings[labels == c] -classwise_mean_features[c])
        #     for c in range(num_classes)
        # ])
        classwise_cov_features = torch.stack([torch.cov(embeddings[labels == c].T) for c in range(num_classes)])
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                    device=classwise_cov_features.device,
                ).unsqueeze(0)
                # gmm = torch.distributions.MultivariateNormal(
                #     loc=classwise_mean_features,
                #     covariance_matrix=(classwise_cov_features + jitter),
                # )
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features,covariance_matrix=(classwise_cov_features+jitter),)
                #MultivariateNormal(loc: torch.Size([10, 2048]), covariance_matrix: torch.Size([10, 2048, 2048]))
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                # if "The parameter covariance_matrix has invalid values" in str(e):
                continue
            break

    return gmm, jitter_eps
