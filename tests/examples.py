import unittest
from unittest import TestCase
from code2seq.utils import compute
from tests import test_utils
import visualization

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

    def test_polyndrome_base(self):
        num_adj_stacks = 4
        graph = nx.Graph()
        # prepare simple graph 0 ->  1 -> 2
        graph.add_nodes_from([
            (0, {'y': 0}),
            (1, {'y': 1}),
            (2, {'y': 1}),
            (3, {'y': 0}),
        ])
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        data = torch_geometric.utils.from_networkx(graph)
        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = False
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)
        stacks_batch = torch.stack([stacks])

        adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=1, bias=False)

        new_stacks = adj_bias_model(stacks_batch)
        visualization.draw_pyg_graph(data)
        visualization.show_matrix(new_stacks[0][0].detach().numpy())
        # visualization.show_matrix(stacks.transpose(0, -1).numpy())
        print('Each cell represent the "color" of an edge between 2 nodes')
        print('Self edges(along the diagonal) catch structural similarity. \n (0,0) == (3,3) and (1,1) == (2,2)')
        print('Symmetries are kept. e.g along the second diagonal, (0,3)==(3,0). (1,2) == (2,1)')
        print(
            f'we would like the edges (0,3):{stacks.transpose(0, -1)[0, 3].tolist()} and (1,2):{stacks.transpose(0, -1)[1, 2].tolist()} to have maximal weight.')


if __name__ == '__main__':
    unittest.main()
