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

    nx.draw(graph, positions, node_color=heatmap, with_labels=True, cmap=plt.cm.Reds)
    nx.draw(graph.subgraph(source_node), positions, node_color=[heatmap[source_node_tensor_index]],
            with_labels=True, font_color='red', cmap=plt.cm.Reds)

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


min_row_size = 1
max_row_size = 10
num_adj_stacks = 5

graph, positions = coloring.graph_generation.create_pyramid(min_row_size, max_row_size)
nx_id_to_tensor_index = {x: i for i, x in enumerate(graph.nodes())}
tensor_id_to_nx_index = {i: x for i, x in enumerate(graph.nodes())}

color_graph(graph)
#node_color=[{'red':'green','green':'red','blue':'blue'}[x] for x in colors]
colors = [index_to_color[graph.nodes[x]['color']] for x in graph.nodes]

nx.draw(graph, positions, node_color=colors, with_labels=True)
#nx.draw(graph,positions,node_color=[{'red':'green','green':'red','blue':'blue'}[x] for x in colors])
plt.show()

data = torch_geometric.utils.from_networkx(graph, all)
stacks = create_stacks(num_adj_stacks)
stacks = stacks.permute(1, 2, 0)
edges_to_nodes_tuple, edges_to_is_same_color = map_tensor_edge_to_networkx_node_ids(graph, stacks)

stacks_batch = torch.stack([stacks.permute(2, 0, 1)])

adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=1, bias=False)

edge_to_relative_edge_weight = {edge: sum(same_color) / len(same_color) for edge, same_color in
                                edges_to_is_same_color.items()}


def mock_edge_weight(edge):
    edges_as_tuples = [tuple(e.numpy()) for e in edge]
    edges_weights = [edge_to_relative_edge_weight[e] for e in edges_as_tuples]
    return torch.tensor(edges_weights).unsqueeze(1)
    # return edge.sum(-1, keepdim=True)
    # should return size n^2x1.. edge dim is n^2 x heads


adj_bias_model.weight = test_utils.MockModule(mock_edge_weight)

new_stacks = adj_bias_model(stacks_batch).squeeze()

edge_string = lambda e1, e2: ["%.3f" % x for x in stacks[nx_id_to_tensor_index[e1], nx_id_to_tensor_index[e2]].tolist()]

fig, axe = plt.subplots(figsize=(12, 7))
title = f'\n(4,2)->(7,5):{edge_string((4, 2), (7, 5))}    ' \
        f'(4,2)->(7,4):{edge_string((4, 2), (7, 4))}\n' \
        f'(8,7)->(9,9):{edge_string((9, 7), (9, 9))}    ' \
        f'(8,7)->(6,6):{edge_string((9, 7), (6, 6))}' \
        f''
axe.set_title(title, loc='right')
nx.draw(graph, positions, node_color=colors, ax=axe, with_labels=True)
plt.show()

source_node = (2, 1)
draw_attention(graph, source_node, new_stacks.softmax(dim=-1))
draw_attention(graph, source_node, new_stacks.softmax(dim=-1) @ (new_stacks))
