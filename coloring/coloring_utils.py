import argparse
import collections

import networkx as nx

from code2seq.utils import compute

torch = compute.get_torch()

from torch_geometric.data import Data

from model.positional.positional_attention_weight import AdjStack

index_to_color = {0: 'red', 1: 'green', 2: 'blue'}


def color_graph(graph):
    color_map = nx.algorithms.greedy_color(graph, 'DSATUR')
    done_colors = [color_map[x] for x in graph]
    assert len(set(done_colors)) == 3, f'expecting to be able to color graph with 3 color, not {set(done_colors)}'

    for (x, color) in color_map.items():
        graph.nodes[x]['color'] = color
    return [index_to_color[graph.nodes[x]['color']] for x in graph.nodes]


# creates same color map but with 0 as white color and "push" the rest of the colors by +1
def index_to_color_map_with_white(index_to_color_map):
    new_color_map = {0: 'gray'}
    for index in index_to_color_map:
        new_color_map[index + 1] = index_to_color_map[index]
    return new_color_map


def create_stacks(data: Data, num_adj_stacks):
    args = argparse.ArgumentParser().parse_args()
    args.adj_stacks = range(num_adj_stacks)
    args.use_distance_bias = False
    stacks = AdjStack(args)(data)['adj_stack']
    return torch.tensor(stacks)


def tensor_to_tuple(tensor):
    return tuple(tensor.numpy())


def map_tensor_edge_to_networkx_node_ids(graph, stacks):
    """maps edge(stack as python tuple of floats) to networkx graph index
    """
    N = len(graph)

    edges_to_nodes_tuple = collections.defaultdict(list)
    for i, node_i in zip(range(N), graph.nodes):
        for j, node_j in zip(range(N), graph.nodes):
            tensor_edge = tuple(stacks[i][j].numpy())
            edges_to_nodes_tuple[tensor_edge].append((node_i, node_j))

    return edges_to_nodes_tuple


def map_tensor_edge_to_color(graph, stacks):
    """gets colored graph.
    return map of  edge(stack as python tuple of floats) to color"""
    N = len(graph)

    edges_to_color = collections.defaultdict(list)
    for i, node_i in zip(range(N), graph.nodes):
        for j, node_j in zip(range(N), graph.nodes):
            tensor_edge = tuple(stacks[i][j].numpy())
            edges_to_color[tensor_edge].append((graph.nodes[node_i]['color'], graph.nodes[node_j]['color']))

    edges_to_is_same_color = {edge: [c_i == c_j for (c_i, c_j) in edge_colors] for edge, edge_colors in
                              edges_to_color.items()}

    return edges_to_is_same_color
