import numpy as np
import torch
from KDEpy import NaiveKDE, TreeKDE
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from torch import nn
from tqdm import tqdm


class KdeModel():

    def __init__(self, kdes, weights):
        self.N = len(kdes)
        self.kdes = kdes
        self.weights = weights

    def log_prob(self, x):
        pdf = []
        for i in range(self.N):
            pdf.append(self.kdes[i].score_samples(x))  # log prob
        pdf = np.stack(pdf, axis=1)
        return pdf

    def prob(self, x):
        return np.sum(self.weights * np.exp(self.log_prob(x)))


def kde_evaluate(net, kde_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader,dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)
            net(data)
            features = net.feature
            logit_B_C = kde_model.log_prob(features.cpu().numpy())

            end = start + len(data)
            logits_N_C[start:end] = torch.from_numpy(logit_B_C)
            labels_N[start:end] = label
            start = end

    return logits_N_C, labels_N


def kde_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        # bandwidths = [0.01, 0.1, 1, 10]
        # grid = GridSearchCV(KernelDensity(kernel='gaussian'), {
        #                     'bandwidth': bandwidths}, verbose=1)
        # grid.fit(embeddings.cpu().numpy())
        # # The best estimated bandwidth density is used as the truth value
        # best_KDEbandwidth = grid.best_params_['bandwidth']
        best_KDEbandwidth = 0.1
        # print(f"best bandwidth:{best_KDEbandwidth}")
        kernel = "gaussian"

        # kdes = [NaiveKDE(kernel='gaussian', bw=1).fit(embeddings[labels == c].cpu().numpy()) for c in range(num_classes)]
        kdes = [KernelDensity(kernel=kernel, bandwidth=best_KDEbandwidth).fit(embeddings[labels == c].cpu().numpy()) for c in range(num_classes)]
        weights = [(labels == c).sum().cpu().numpy() / len(labels) for c in range(num_classes)]

    return KdeModel(kdes, weights)
