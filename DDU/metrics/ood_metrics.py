# Utility functions to get OOD detection ROC curves and AUROC scores
# Ideally should be agnostic of model architectures

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from utils.ensemble_utils import ensemble_forward_pass
from metrics.classification_metrics import get_logits_labels
from metrics.uncertainty_confidence import entropy, logsumexp,confidence
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from matplotlib import pyplot as plt 

def plot_roc(preds, labels, title="Receiver operating characteristic"):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_pr(preds, labels, title="Precision recall curve"):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.  
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.          
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def auroc(preds, labels, pos_label=1):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.  
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    return auc(fpr, tpr)


def auprc(preds, labels, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label. 
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class         
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class        
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)



def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, conf=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, conf=conf)


def get_roc_auc_logits(logits, ood_logits, uncertainty, device, conf=False):
    uncertainties = uncertainty(logits)#logits: torch.Size([10000, 10])-> uncertainties:torch.Size([10000]
    ood_uncertainties = uncertainty(ood_logits)

    # In-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
    # OOD
    bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))
    if conf: #True时，指标越高信心越高，这时将in样本置为1，False时，指标越高信心越低，这时将ood样本置为1
        bin_labels = 1 - bin_labels
    bin_labels = bin_labels.cpu().numpy()

    in_scores = uncertainties
    ood_scores = ood_uncertainties
    scores = torch.cat((in_scores, ood_scores)).cpu().numpy()
    scores = np.nan_to_num(scores)#处理nan

    fpr, tpr, thresholds = metrics.roc_curve(bin_labels, scores)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels, scores)
    auroc = metrics.roc_auc_score(bin_labels, scores)
    auprc = metrics.average_precision_score(bin_labels, scores)

    # auroc2 = metrics.auc(fpr, tpr)  # the value of roc_auc1
    # aupr2 = metrics.auc(recall, precision)  # the value of roc_auc1

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc


def get_roc_auc_ensemble(model_ensemble, test_loader, ood_test_loader, uncertainty, device):
    bin_labels_uncertainties = None
    uncertainties = None

    for model in model_ensemble:
        model.eval()

    bin_labels_uncertainties = []
    uncertainties = []
    with torch.no_grad():
        # Getting uncertainties for in-distribution data
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.zeros(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        # Getting entropies for OOD data
        for data, label in ood_test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.ones(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        bin_labels_uncertainties = torch.cat(bin_labels_uncertainties)
        uncertainties = torch.cat(uncertainties)

    fpr, tpr, roc_thresholds = metrics.roc_curve(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy()
    )
    auroc = metrics.roc_auc_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())

    return (fpr, tpr, roc_thresholds), (precision, recall, prc_thresholds), auroc, auprc
