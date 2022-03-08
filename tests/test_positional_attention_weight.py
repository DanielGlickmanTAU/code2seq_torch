import unittest
from unittest import TestCase
from code2seq.utils import compute

torch = compute.get_torch()
import argparse

import networkx as nx
import torch_geometric

from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights


class TestAdjStack(TestCase):
    def _example_graph(self):
        graph = nx.Graph()
        # prepare simple graph 0 ->  1 -> 2 -> 0 2->3 3->4->5->6->3
        graph.add_edges_from([(0, 1), (1, 2), (2, 0),
                              (2, 3),
                              (3, 4), (4, 5), (5, 6), (6, 3)
                              ])
        data = torch_geometric.utils.from_networkx(graph)
        return data

    def test_add_args(self):
        num_adj_stacks = 3
        data = self._example_graph()
        args = argparse.ArgumentParser().parse_args()
        args.use_distance_bias = False
        args.adj_stacks = range(num_adj_stacks)
        stacks = AdjStack(args)(data)['adj_stack']

        distance = 2
        self.assertAlmostEqual(stacks[distance][0][3], 1 / 6, delta=0.01)
        self.assertAlmostEqual(stacks[distance, 0, 2], 1 / 4, delta=0.01)

    def test_positional_weight_dims(self):
        num_adj_stacks = 3
        data = self._example_graph()
        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = False
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)
        stacks_batch = torch.stack([stacks, stacks + torch.rand(1)])

        adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=num_adj_stacks, bias=False)
        with torch.no_grad():
            adj_bias_model.weight.weight.data = torch.eye(num_adj_stacks)

        new_stacks = adj_bias_model(stacks_batch)
        self.assertTrue((new_stacks == stacks_batch).all())

    def test_use_distance_bias(self):
        num_adj_stacks = 3
        graph = nx.Graph()
        # prepare simple graph 0 ->  1 -> 2
        graph.add_edges_from([(0, 1), (1, 2)])
        data = torch_geometric.utils.from_networkx(graph)
        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = True
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)
        stacks_batch = torch.stack([stacks])

        adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=1, bias=False)
        with torch.no_grad():
            adj_bias_model.weight.weight.data = torch.tensor([[0., 1., 2.]])

        expected_dist_bias = torch.tensor(
            [  # nodes:0   1   2
                [0, 1, 2],
                [1, 0, 1],
                [2, 1, 0]
            ]
        )
        new_stacks = adj_bias_model(stacks_batch)
        self.assertTrue((new_stacks == expected_dist_bias).all())


if __name__ == '__main__':
    unittest.main()
