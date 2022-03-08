import argparse
from typing import Union, List

from torch.nn import Sequential
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from gnn import GNN


def as_pyg_batch(dataset: Union[Dataset, List[Data]], batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    return list(loader)[0]


def get_args_with_adj_stack(list_of_stacks: Union[List[int], int]):
    args = argparse.ArgumentParser().parse_args()
    args.use_distance_bias = False
    args.adj_stacks = list_of_stacks if isinstance(list_of_stacks, list) else range(list_of_stacks)
    return args


def get_positional_biases(model: GNN) -> list:
    return [transformer_layer.attention_layer.positional_bias for transformer_layer in
            model.gnn_transformer.transformer.layers]

#return list of [(linear1, linear2)] for each attention layer
def get_feedforward_layers(model:GNN) -> list:
    return [(transformer_layer.linear1,transformer_layer.linear2)
            for transformer_layer in model.gnn_transformer.transformer.layers]


def apply_sequential_and_get_intermediate_results(model:Sequential,input):
    out = []
    x = input
    for layer in model:
        x = layer(x)
        out.append(x)
    return out
