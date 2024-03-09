"""
This module contains utility code for evaluating a model.
"""

import torch
from metrics.uncertainty_confidence import entropy
from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_ensemble,
)
from metrics.calibration_metrics import expected_calibration_error
from metrics.ood_metrics import get_roc_auc, get_roc_auc_ensemble


def get_eval_stats(net, test_loader, ood_test_loader, device):
    """
    Util method for getting evaluation measures taken during training time.
    """
    conf_matrix, accuracy, labels, predictions, confidences = test_classification_net(net, test_loader, device)
    ece = expected_calibration_error(confidences, predictions, labels, num_bins=15)
    (_, _, _), (_, _, _), auroc, auprc = get_roc_auc(net, test_loader, ood_test_loader, entropy, device)
    return accuracy, ece, auroc, auprc


def get_eval_stats_ensemble(net_ensemble, test_loader, ood_test_loader, device):
    """
    Util method for getting evaluation measures taken during training time for an ensemble.
    """
    (conf_matrix, accuracy, labels, predictions, confidences,) = test_classification_net_ensemble(
        net_ensemble, test_loader, device
    )
    ece = expected_calibration_error(confidences, predictions, labels, num_bins=15)
    (_, _, _), (_, _, _), auroc, auprc = get_roc_auc_ensemble(
        net_ensemble, test_loader, ood_test_loader, entropy, device
    )
    return accuracy, ece, auroc, auprc



def model_load_name(model_name, sn, mod, coeff, seed, contrastive):
    if sn:
        if mod:
            strn = "_sn_" + str(coeff) + "_mod_"
        else:
            strn = "_sn_" + str(coeff) + "_"
    else:
        if mod:
            strn = "_mod_"
        else:
            strn = "_"

    if contrastive:
        return str(model_name) + strn + "seed_" + str(seed) + "_contrastive"
    else:
        return str(model_name) + strn + "seed_" + str(seed)




def accuracy(output: torch.TensorType,
             target: torch.TensorType,
             topk: tuple = (1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0000 / batch_size))
        return res