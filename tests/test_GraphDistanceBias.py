from unittest import TestCase
import argparse

import torch
import networkx as nx
import torch_geometric

from GraphDistanceBias import GraphDistanceBias
from dataset_transformations import DistanceCalculator, unconnected


class TestGraphDistanceBias(TestCase):
    def test_embedding_distance(self):
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
        # set const embedding so we can test
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

    def test_embedding_distance(self):
        args = argparse.ArgumentParser().parse_args()
        args.max_graph_dist = 2
        args.distance_bias = True
        num_heads = 3

        receptive_field = [1, 2, unconnected]
        distance_bias = GraphDistanceBias(args, receptive_fields=receptive_field, num_heads=num_heads)
        graph = nx.Graph()
        # prepare simple graph 0 -> 1 -> 2 -> 3
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        data = torch_geometric.utils.from_networkx(graph)

        # add distances
        data = DistanceCalculator()(data)
        random_const = 42
        # set const embedding so we can test
        with torch.no_grad():
            distance_bias.distance_embedding.weight[0] = 0 * random_const
            distance_bias.distance_embedding.weight[1] = 1 * random_const
            distance_bias.distance_embedding.weight[2] = 2 * random_const
            distance_bias.distance_embedding.weight[3] = 3 * random_const

        bias = distance_bias._embed_distances(data.distances, 'cpu')
        for head_num in range(num_heads):
            head_bias = bias[head_num]
            for i in [0, 1, 2, 3]:
                for j in [0, 1, 2, 3]:
                    receptive = receptive_field[head_num]
                    if abs(i - j) <= receptive:
                        assert head_bias[i][j] == min(args.max_graph_dist, abs(i - j)) * random_const
                    else:
                        assert head_bias[i][j] == float('-inf')
