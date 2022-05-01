import collections

import tabulate

import coloring.graph_generation
from code2seq.utils import compute

torch = compute.get_torch()
import argparse

import torch_geometric

from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights
from tests import examples, test_utils
import networkx as nx
import matplotlib.pyplot as plt
import pandas

index_to_color = {0: 'red', 1: 'green', 2: 'blue'}


def color_graph(graph):
    color_map = nx.algorithms.greedy_color(graph, 'DSATUR')
    done_colors = [color_map[x] for x in graph]
    assert len(set(done_colors)) == 3, f'expecting to be able to color graph with 3 color, not {set(done_colors)}'

    for (x, color) in color_map.items():
        graph.nodes[x]['color'] = color


def draw_attention(graph, source_node, attention_matrix):
    nx_id_to_tensor_index = {x: i for i, x in enumerate(graph.nodes())}
    tensor_id_to_nx_index = {i: x for i, x in enumerate(graph.nodes())}
    assert attention_matrix.dim() == 2 and attention_matrix.shape[0] == attention_matrix.shape[1]

    source_node_tensor_index = nx_id_to_tensor_index[source_node]
    source_node_attention_scores = attention_matrix[source_node_tensor_index]
    source_node_attention_scores_nx_indexed = {tensor_id_to_nx_index[i]: score.item() for i, score in
                                               enumerate(source_node_attention_scores)}
    heatmap = [source_node_attention_scores_nx_indexed[n] for n in graph.nodes]
    nx.draw(graph, positions, node_color=heatmap, with_labels=True)
    nx.draw(graph.subgraph(source_node), positions, node_color=[heatmap[source_node_tensor_index]],
            with_labels=True, font_color='red')
    plt.show()


def create_stacks(num_adj_stacks):
    args = argparse.ArgumentParser().parse_args()
    args.adj_stacks = range(num_adj_stacks)
    args.use_distance_bias = False
    stacks = AdjStack(args)(data)['adj_stack']
    return torch.tensor(stacks)


def map_tensor_edge_to_networkx_node_ids(graph, edges):
    N = edges.shape[0]

    edges_to_nodes_tuple = collections.defaultdict(list)
    edges_to_color = collections.defaultdict(list)
    for i, node_i in zip(range(N), graph.nodes):
        for j, node_j in zip(range(N), graph.nodes):
            tensor_edge = tuple(stacks[i][j].numpy())
            edges_to_nodes_tuple[tensor_edge].append((node_i, node_j))
            edges_to_color[tensor_edge].append((graph.nodes[node_i]['color'], graph.nodes[node_j]['color']))

    edges_to_is_same_color = {edge: [c_i == c_j for (c_i, c_j) in edge_colors] for edge, edge_colors in
                              edges_to_color.items()}

    return edges_to_nodes_tuple, edges_to_is_same_color


def calc_stats(edges_to_nodes_tuple, edges_to_is_same_color):
    edge_size = len(list(edges_to_nodes_tuple.keys())[0])
    zero_edge = tuple([0. for i in range(edge_size)])

    num_uncovered_pairs = len(edges_to_nodes_tuple[zero_edge])
    num_all_pairs = sum([len(x) for x in edges_to_nodes_tuple.values()])
    reachable_percent = 1. - num_uncovered_pairs / num_all_pairs

    edges_that_connect_same_color_and_also_different_color = [k for k, v in edges_to_is_same_color.items() if
                                                              len(set(v)) > 1 if k != zero_edge]
    num_unique_edges = len(edges_to_nodes_tuple)
    ambigious_edges = len(edges_that_connect_same_color_and_also_different_color)
    print(f'num unique edges: {num_unique_edges}')
    print(
        f'num edges that exist between both nodes of same color and different color: {ambigious_edges}')
    print(f'reachble percent {reachable_percent}')

    return num_unique_edges, ambigious_edges, reachable_percent


min_row_size = 1

stats = {}

for max_row_size in [5]:
    graph, positions = coloring.graph_generation.create_pyramid(min_row_size, max_row_size)
    color_graph(graph)
    # for num_adj_stacks in range(max_row_size - 1, max_row_size + 10):
    for num_adj_stacks in [max_row_size]:
        colors = [index_to_color[graph.nodes[x]['color']] for x in graph.nodes]
        # nx.draw(graph, positions, node_color=colors, with_labels=True)
        # plt.show()

        data = torch_geometric.utils.from_networkx(graph, all)
        stacks = create_stacks(num_adj_stacks)
        stacks = stacks.permute(1, 2, 0)
        edges_to_nodes_tuple, edges_to_is_same_color = map_tensor_edge_to_networkx_node_ids(graph, stacks)
        num_unique_edges, ambigious_edges, reachable_percent = calc_stats(edges_to_nodes_tuple, edges_to_is_same_color)
        stats[(max_row_size, num_adj_stacks)] = (num_unique_edges, ambigious_edges, reachable_percent)

stacks_batch = torch.stack([stacks.permute(2, 0, 1)])
print('a')
table = [
    ['pyramid base', 'num adj stacks', 'unique edges', 'of which ambiguous', '% pairs in receptive field']

]
for pyramid_base, edge_dim in stats:
    unique_edges, ambiguous, p_pairs_in_receptive = stats[(pyramid_base, edge_dim)]
    table.append([pyramid_base, edge_dim, unique_edges, ambiguous, p_pairs_in_receptive])

df = pandas.DataFrame(table[1:], columns=table[0])
print(tabulate.tabulate(table))
