from unittest import TestCase
from code2seq.utils import compute

torch = compute.get_torch()
import argparse

import networkx as nx
import torch_geometric

from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights


class TestAdjStack(TestCase):
    def test_add_args(self):
        num_adj_stacks = 3
        num_heads = 4
        graph = nx.Graph()
        # prepare simple graph 0 ->  1 -> 2 -> 0
        graph.add_edges_from([(0, 1), (1, 2), (2, 0),
                              (2, 3),
                              (3, 4), (4, 5), (5, 6), (6, 3)
                              ])
        data = torch_geometric.utils.from_networkx(graph)
        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        stacks = AdjStack(args)(data)['adj_stack']

        self.assertAlmostEqual(stacks[2][0][3], 1 / 6, delta=0.01)

        # stacks_batch = torch.tensor(stacks).unsqueeze(0)
        # adj_bias_model = AdjStackAttentionWeights(num_adj_stacks=num_adj_stacks, num_heads=num_heads)
        # positional_attention = adj_bias_model(stacks_batch)
        # print(positional_attention)
