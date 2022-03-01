import torch
from torch import nn

n_classes = 2

r"""
taken from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/SBMs_node_classification/gin_net.py
"""


def sbm_loss(pred, label):
    # calculating label weights for weighted loss computation
    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(n_classes).long().to(pred.device)
    cluster_sizes[torch.unique(label)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()

    # weighted cross-entropy for unbalanced classes
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, label)

    return loss
