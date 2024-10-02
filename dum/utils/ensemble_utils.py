"""
Utilities for processing a deep ensemble.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import glob
from net.vgg import vgg16
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn

from metrics.uncertainty_confidence import entropy_prob, mutual_information_prob
from utils.attack_utils import fgsm_attack, bim_attack, deepfool_attack, pgd_attack, cw_attack
import warnings

from functools import partial

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


def load_ensemble(ensemble_loc, model_name, device, num_classes=10, ensemble_len=5, num_epochs=350, seed=1, **kwargs):
    ensemble = []
    cudnn.benchmark = True
    files = glob.glob(f"{ensemble_loc}/{model_name}*.model")
    for i in range(ensemble_len):
        net = models[model_name](num_classes=num_classes, temp=1.0, **kwargs).to(device)
        net.load_state_dict(torch.load(files[i]))
        ensemble.append(net)
    return ensemble


def ensemble_forward_pass(model_ensemble, data):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
    outputs = []
    for i, model in enumerate(model_ensemble):
        output = F.softmax(model(data), dim=1)
        outputs.append(torch.unsqueeze(output, dim=0))

    outputs = torch.cat(outputs, dim=0)
    mean_output = torch.mean(outputs, dim=0)
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)

    return mean_output, predictive_entropy, mut_info


def ensemble_forward_pass_adv(model_ensemble, images, device, perturbation="fgsm"):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
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

    outputs = []

    images.requires_grad = True  #images.required_grad区分,用required_grad梯度为None
    net0 = model_ensemble[0]
    logits = net0(images)
    _, pred = torch.max(logits, 1)
    images_adv = perturb(net0, images, pred, device)
    # images_adv.requires_grad = True  #images.required_grad区分,用required_grad梯度为None

    for i, model in enumerate(model_ensemble):
        output = F.softmax(model(images_adv), dim=1)
        outputs.append(torch.unsqueeze(output, dim=0))

    outputs = torch.cat(outputs, dim=0)
    mean_output = torch.mean(outputs, dim=0)
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)

    return mean_output, predictive_entropy, mut_info
