import numpy
import torch as torch
from torch_geometric.loader import DataLoader

from args_parse import get_default_args
from code2seq.utils import compute
from data import dataloader_utils
from ogb.graphproppred import PygGraphPropPredDataset

torch = compute.get_torch()
import argparse
import torch
import torch_geometric

from model.positional.positional_attention_weight import AdjStack
from tests import examples, test_utils
import networkx as nx
import matplotlib.pyplot as plt
import tabulate


def create_stacks(num_adj_stacks):
    args = argparse.ArgumentParser().parse_args()
    args.adj_stacks = range(num_adj_stacks)
    args.use_distance_bias = False
    stacks = AdjStack(args)(data)['adj_stack']
    return torch.tensor(stacks)


min_row_size = 1
max_row_size = 20
num_adj_stacks = 5

# graph, positions = examples.create_pyramid(max_row_size, min_row_size)
# data = torch_geometric.utils.from_networkx(graph)
# stacks = create_stacks(num_adj_stacks)
# stacks = stacks.permute(1, 2, 0)
dataset_samples = 1000
args = get_default_args()
args.dataset = "ogbg-molhiv"
args.adj_stacks = [*range(num_adj_stacks)]
data = PygGraphPropPredDataset(name=args.dataset,
                               transform=AdjStack(args))

data = list(DataLoader(data, batch_size=dataset_samples))[0]
list_of_graph_stacks = [y.transpose(1, 2, 0).reshape(-1, num_adj_stacks) for y in data['adj_stack']]
stacks = torch.tensor(numpy.concatenate(list_of_graph_stacks))

d = stacks.shape[-1]
stacks = stacks.view(-1, d)
mean, std = stacks.mean(dim=0), stacks.std(dim=0)
table = [
    ['distance', 'mean', 'std'],
]
for i in range(d):
    table.append([i, mean[i].item(), std[i].item()])
print(tabulate.tabulate(table))

dist_to_plot = 4
plt.hist(stacks[:, dist_to_plot].tolist())
