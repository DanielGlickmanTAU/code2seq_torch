import numpy

from code2seq.utils import compute

torch = compute.get_torch()
import itertools

import networkx as nx
import torch_geometric
from torch.utils.data import Dataset


def _create_clique(n):
    G = nx.Graph()
    edges = itertools.combinations(range(0, n), 2)
    G.add_edges_from(edges)
    return G


def Clique(n):
    graph = _create_clique(n)
    graph.name = f'{n}_clique'
    for node in graph.nodes:
        graph.nodes[node]['color'] = graph.name
    return graph


def Cycle(n):
    graph = nx.cycle_graph(n)
    graph.name = f'{n}_cycle'
    for node in graph.nodes:
        graph.nodes[node]['color'] = graph.name
    return graph


def HourGlass():
    graph = nx.Graph()
    graph.name = 'hg'
    graph.add_edges_from(
        [(0, 1), (1, 2), (2, 0),
         (5, 3), (3, 4), (4, 5),
         (0, 3)
         ]
    )
    return graph


def SatGraph():
    graph = nx.Graph()
    # return 5_clique - 5_cycle..


def JoinedSquared():
    graph = nx.Graph()
    graph.name = 'jsquare'
    graph.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 0),
         (1, 4), (4, 5), (5, 2)]
    )
    return graph


cycle_4 = Cycle(4)
cycle_5 = Cycle(5)
clique_4 = Clique(4)
clique_5 = Clique(5)



class WordGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        self.name_2_label = {graph.name: i for i, graph in enumerate(graphs)}
        self.label_2_name = {i: graph.name for i, graph in enumerate(graphs)}
        self.dataset = []
        for graph in graphs:
            pyg_graph = create_pyg_graph(graph, self.name_2_label)
            self.dataset.append(pyg_graph)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class WordsCombinationGraphDataset(Dataset):
    def __init__(self, word_graphs, num_samples, words_per_sample):
        self.word_graphs = word_graphs
        self.name_2_label = {graph.name: i for i, graph in enumerate(word_graphs)}
        self.label_2_name = {i: graph.name for i, graph in enumerate(word_graphs)}
        self.dataset = []

        for i in range(num_samples):
            selected_words = numpy.random.choice(word_graphs, words_per_sample).tolist()
            graph = join_graphs(selected_words)
            pyg_graph = create_pyg_graph(graph, self.name_2_label)
            self.dataset.append(pyg_graph)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_pyg_graph(graph, name_2_label):
    pyg_graph = torch_geometric.utils.from_networkx(graph)
    pyg_graph.x = torch.zeros((pyg_graph.num_nodes,), dtype=torch.long)

    pyg_graph.y = torch.tensor([name_2_label[attr['color']] for _, attr in graph.nodes(data=True)])

    N = len(graph.nodes)
    # graph.positions = {i: (i if i < N / 2 else N - i, 1 if i < N / 2 else -1) for i, x in enumerate(
    #     graph.nodes)}
    graph.positions = None
    pyg_graph.graph = graph
    return pyg_graph


def merge_graphs(left_graph, right_graph):
    new_graph = nx.disjoint_union(left_graph, right_graph)
    left_graph_end_edge, right_graph_start_edge = len(left_graph) - 1, len(left_graph)
    new_graph.add_edge(left_graph_end_edge, right_graph_start_edge)

    # 4) todo: offset right graph positions.. in test flow no position
    return new_graph


def join_graphs(graphs):
    left_graph = graphs[0]
    for right_graph in graphs[1:]:
        left_graph = merge_graphs(left_graph, right_graph)
    return left_graph
