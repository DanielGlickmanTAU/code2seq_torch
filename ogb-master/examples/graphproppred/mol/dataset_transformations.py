from dataclasses import dataclass

import torch_geometric.transforms
from torch_geometric.data import Data

import graph_algos
import torch

unconnected = 99999


class DistanceCalculator(torch_geometric.transforms.BaseTransform):
    def __call__(self, data: Data):
        edge_index = data.edge_index
        # N = data.x.size(0)
        N = data.num_nodes
        # (row, col) = data.edge_index
        adj = torch.full([N, N], unconnected)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        adj.fill_diagonal_(0)
        shortest_path = graph_algos.floyd_warshall(adj)
        # pytorch-geometric collate expects tensor with all same dims, except for the 0 dim. so we pack here to a single dim, and unpack it back in GNN#forward
        # data.distances = shortest_path.view(-1)
        data.distances = shortest_path

        return data


class AdjStack(torch_geometric.transforms.BaseTransform):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--adj_powers', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                            help='list of powers to raise and stack the adj matrix.')

    def __init__(self, args):
        self.adj_powers = args.adj_powers
        assert len(self.adj_powers) == len(set(self.adj_powers)), f'duplicate power in {self.adj_powers}'

    def __call__(self, data: Data):
        edge_index = data.edge_index
        # N = data.x.size(0)
        N = data.num_nodes
        # (row, col) = data.edge_index
        adj = torch.full([N, N], 0)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        adj.fill_diagonal_(0)

        adj_stack = torch.stack([torch.matrix_power(adj, exp) for exp in self.adj_powers])
        # need this for now
        adj_stack = adj_stack.numpy()
        data.adj_stack = adj_stack

        return data
