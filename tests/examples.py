import random
import unittest
from unittest import TestCase
from code2seq.utils import compute
from tests import test_utils
import visualization
import matplotlib.pyplot as plt

torch = compute.get_torch()
import argparse

import networkx as nx
import torch_geometric
from matplotlib.pyplot import cm

from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights


class TestAdjStack(TestCase):

    def test_polyndrome_base(self):
        num_adj_stacks = 4
        graph = nx.Graph()
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

        adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=3, bias=False)
        adj_bias_model.weight = test_utils.MockModule(lambda x: x.sum(-1, keepdim=True))

        new_stacks = adj_bias_model(stacks_batch)
        visualization.draw_pyg_graph(data)
        # visualization.show_matrix(new_stacks[0][0].detach().numpy())
        # visualization.show_matrix(new_stacks[0].permute(1, 2, 0).mean(dim=-1).detach().numpy())
        visualization.show_matrix(new_stacks[0][0].softmax(dim=-1).detach().numpy())
        edges = stacks.transpose(0, -1)
        print('Each cell represent the "color" of an edge between 2 nodes')
        print('Self edges(along the diagonal) catch structural similarity. \n (0,0) == (3,3) and (1,1) == (2,2)')
        print('Symmetries are kept. e.g along the second diagonal, (0,3)==(3,0). (1,2) == (2,1)')
        print(
            f'we would like the edges (0,3):{edges[0, 3].tolist()} and (1,2):{edges[1, 2].tolist()} to have maximal weight.')
        print(
            f'and the edges (0,1):{edges[0, 1].tolist()}, (0,2):{edges[0, 2].tolist()} and (1,2):{edges[1, 2].tolist()} to have minimal weight')

    def test_polyndrome_edge_attention(self):
        num_adj_stacks = 5
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
        visualization.draw_pyg_graph(data)

        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = False
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)
        stacks = stacks.permute(1, 2, 0)

        N = stacks.shape[0]
        F = stacks.shape[2]  # == num_adj_stacks
        Wq = torch.rand(F, F)
        Wk = -Wq.flip(0)

        Q = (stacks @ Wq).reshape(N, N * F)
        K = (stacks @ Wk).reshape(N, N * F)

        att = Q @ K.T
        att = att.softmax(dim=-1)
        print(att)
        visualization.show_matrix(att, cmap=cm.Reds)

    def test_directed_graph(self):
        num_adj_stacks = 3
        graph = nx.DiGraph()
        # prepare simple graph 0 ->  1 -> 2
        graph.add_nodes_from([
            (0, {'y': 0}),
            (1, {'y': 0}),
            (2, {'y': 1}),
            (3, {'y': 1}),
            (4, {'y': 1}),
        ])
        graph.add_edges_from([
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
        ]
        )
        data = torch_geometric.utils.from_networkx(graph)
        visualization.draw_pyg_graph(data, to_undirected=False)

        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = False
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)
        stacks = stacks.permute(1, 2, 0)

        N = stacks.shape[0]
        F = stacks.shape[2]  # == num_adj_stacks
        Wq = torch.eye(F)
        Wk = Wq

        Q = (stacks @ Wq).reshape(N, N * F)
        K = (stacks @ Wk).reshape(N, N * F)

        att = Q @ K.T
        att = att.softmax(dim=-1)
        print(att)
        visualization.show_matrix(stacks, text='Nodes Edge Features')
        visualization.show_matrix(att, cmap=cm.Reds, text='Nodes edge based similarity')

    def test_2_color(self):
        graph = nx.Graph()
        graph.add_nodes_from([
            (0, {'y': 0}),
            (1, {'y': -1}),
            (2, {'y': -1}),
            (3, {'y': -1}),
            (4, {'y': -1}),
            (5, {'y': 1}),
        ])
        graph.add_edges_from([
            (0, 3), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5)
        ]
        )
        visualization.draw_pyg_graph(graph)

        graph.remove_edges_from(graph.edges)
        graph.add_edges_from([
            (0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)
        ])

        visualization.draw_pyg_graph(graph)

    def test_long_line(self):
        num_nodes = 10
        num_adj_stacks = 3
        graph = nx.Graph()
        graph.add_nodes_from([
            (i, {'y': 0 if i == 0 else -1}) for i in range(num_nodes)])
        graph.add_edges_from([
            (i, i + 1) for i in range(num_nodes - 1)
        ]
        )
        visualization.draw_pyg_graph(graph)

        data = torch_geometric.utils.from_networkx(graph)

        args = argparse.ArgumentParser().parse_args()
        args.adj_stacks = range(num_adj_stacks)
        args.use_distance_bias = True
        stacks = AdjStack(args)(data)['adj_stack']
        stacks = torch.tensor(stacks)

        # zero out distance 0 and distance 1 stack
        stacks[:-1] = 0

        stacks_batch = torch.stack([stacks])
        adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=3, bias=False)
        adj_bias_model.weight = test_utils.MockModule(lambda x: x.sum(-1, keepdim=True))

        new_stacks = adj_bias_model(stacks_batch)
        print(new_stacks)

        stacks = stacks.permute(1, 2, 0)
        N = stacks.shape[0]
        F = stacks.shape[2]  # == num_adj_stacks
        Wq = torch.rand(F, F)
        Wk = -Wq.flip(0)

        Q = (stacks @ Wq).reshape(N, N * F)
        K = (stacks @ Wk).reshape(N, N * F)

        att = Q @ K.T
        att = att.softmax(dim=-1)
        print(att)
        # visualization.show_matrix(att, cmap=cm.Reds)

    def test_hexagon_graph(self):
        min_row_size = 4
        max_row_size = 6
        graph = create_hexagon_from_triangles(max_row_size, min_row_size)

        positions = {(row, col): (col + 0.5 * abs(row - max_row_size), -row) for (row, col) in graph.nodes}
        colors = ['gray' for x in graph.nodes]
        blue_index, red_index = random.sample(range(len(colors)), k=2)
        colors[blue_index] = 'blue'
        colors[red_index] = 'red'
        nx.draw(graph, positions, node_color=colors)
        plt.show()

        colors = [['red', 'blue', 'green'][i] for i in
                  nx.algorithms.greedy_color(graph).values()]
        nx.draw(graph, positions, node_color=colors, with_labels=True)
        plt.show()

        print('as')


def create_hexagon_from_triangles(max_row_size, min_row_size):
    graph = TestAdjStack.create_pyramid(max_row_size, min_row_size)
    # lower graph part:
    for row in range(max_row_size + 1, 2 * max_row_size - min_row_size + 1):
        row_size = 2 * max_row_size - row
        for col in range(1, row_size):
            graph.add_edge((row, col), (row, col - 1))
        for col in range(row_size):
            graph.add_edge((row, col), (row - 1, col))
            graph.add_edge((row, col), (row - 1, col + 1))
    return graph


def create_pyramid(max_row_size, min_row_size):
    graph = nx.Graph()
    # +1 here just makes it max row size correct(and not -1)
    for row in range(min_row_size, max_row_size + 1):
        row_index = row - min_row_size
        for col in range(1, row):
            graph.add_edge((row_index, col), (row_index, col - 1))
    for row in range(min_row_size, max_row_size):
        row_index = row - min_row_size
        for col in range(row):
            graph.add_edge((row_index, col), (row_index + 1, col))
            graph.add_edge((row_index, col), (row_index + 1, col + 1))

    positions = {(row, col): (col + 0.5 * abs(row - max_row_size), -row) for (row, col) in graph.nodes}
    return graph, positions


if __name__ == '__main__':
    unittest.main()
