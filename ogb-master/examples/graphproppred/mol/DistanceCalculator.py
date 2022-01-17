import torch_geometric.transforms
from torch_geometric.data import Data

import graph_algos
import torch

unconnected = 9999


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
