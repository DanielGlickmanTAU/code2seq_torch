import argparse
from typing import Union, List

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


def as_pyg_batch(dataset: Union[Dataset, List[Data]], batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    return list(loader)[0]


def get_args_with_adj_stack(list_of_stacks: Union[List[int], int]):
    args = argparse.ArgumentParser().parse_args()
    args.use_distance_bias = False
    args.adj_stacks = list_of_stacks if isinstance(list_of_stacks, list) else range(list_of_stacks)
    return args
