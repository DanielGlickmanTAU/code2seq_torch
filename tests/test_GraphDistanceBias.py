from unittest import TestCase

import torch
from torch_geometric.data import Batch

from DistanceCalculator import DistanceCalculator
from GraphDistanceBias import GraphDistanceBias
import argparse
import networkx as nx
import torch_geometric


class TestGraphDistanceBias(TestCase):
    def test_masking_properly(self):
        args = argparse.ArgumentParser().parse_args()
        args.max_graph_dist = 10
        args.distance_bias = True
        num_heads = 2

        distance_bias = GraphDistanceBias(args, num_heads=num_heads)
        graph = nx.Graph()
        # prepare simple graph 0 -> 1 -> 2 -> 3
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        data = torch_geometric.utils.from_networkx(graph)

        # add distances
        data = DistanceCalculator()(data)
        random_const = 42
        #set const embedding so we can test
        with torch.no_grad():
            distance_bias.distance_embedding.weight[0] = 0 * random_const
            distance_bias.distance_embedding.weight[1] = 1 * random_const
            distance_bias.distance_embedding.weight[2] = 2 * random_const
            distance_bias.distance_embedding.weight[3] = 3 * random_const

        bias = distance_bias._embed_distances(data.distances, 'cpu')
        for head_num in range(num_heads):
            head_bias = bias[head_num]
            for i in [1, 2, 3]:
                for j in [1, 2, 3]:
                    assert head_bias[i][j] == abs(i - j) * random_const
