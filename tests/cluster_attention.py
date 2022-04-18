import collections

import tabulate

from code2seq.utils import compute

torch = compute.get_torch()
import argparse

import torch_geometric

from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights
from tests import examples, test_utils
import networkx as nx
import matplotlib.pyplot as plt


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
    for i, node_i in zip(range(N), graph.nodes):
        for j, node_j in zip(range(N), graph.nodes):
            tensor_edge = tuple(stacks[i][j].numpy())
            edges_to_nodes_tuple[tensor_edge].append((node_i, node_j))

    return edges_to_nodes_tuple


num_adj_stacks = 5

graph = nx.Graph()
graph.add_edges_from(
    [
        (0, 1), (1, 2), (2, 0),
        (2, 3), (2, 4), (2, 5),
        (4, 5)
    ]

)
nx_id_to_tensor_index = {x: i for i, x in enumerate(graph.nodes())}
tensor_id_to_nx_index = {i: x for i, x in enumerate(graph.nodes())}

nx.draw(graph, with_labels=True)
plt.show()

data = torch_geometric.utils.from_networkx(graph)
stacks = create_stacks(num_adj_stacks)
stacks = stacks.permute(1, 2, 0)
edges_to_nodes_tuple = map_tensor_edge_to_networkx_node_ids(graph, stacks)

stacks_batch = torch.stack([stacks.permute(2, 0, 1)])

adj_bias_model = AdjStackAttentionWeights(num_adj_stacks, num_heads=1, bias=False)


def mock_edge_weight(edge):
    edges_as_tuples = [tuple(e.numpy()) for e in edge]
    edges_weights = [edge_to_relative_edge_weight[e] for e in edges_as_tuples]
    return torch.tensor(edges_weights).unsqueeze(1)
    # return edge.sum(-1, keepdim=True)
    # should return size n^2x1.. edge dim is n^2 x heads


adj_bias_model.weight = test_utils.MockModule(mock_edge_weight)

new_stacks = adj_bias_model(stacks_batch).squeeze()

source_node = 2
draw_attention(graph, source_node, new_stacks.softmax(dim=-1))
draw_attention(graph, source_node, new_stacks.softmax(dim=-1) @ (new_stacks))
stacks.reshape(6,-1).numpy()