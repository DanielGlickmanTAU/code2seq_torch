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
    return graph


def Cycle(n):
    graph = nx.cycle_graph(n)
    graph.name = f'{n}_cycle'
    return graph


cycle_4 = Cycle(4)
cycle_5 = Cycle(5)
clique_4 = Clique(4)
clique_5 = Clique(5)

basic_graphs = [cycle_4, cycle_5, clique_4, clique_5]
name_2_label = {graph.name: i for i, graph in enumerate(basic_graphs)}
label_2_name = {i: graph.name for i, graph in enumerate(basic_graphs)}


class WordGraphDataset(Dataset):
    def __init__(self):
        self.name_2_label= name_2_label
        self.dataset = []
        for graph in basic_graphs:
            pyg_graph = self.create_pyg_graph(graph)
            self.dataset.append(pyg_graph)

    def create_pyg_graph(self, graph):
        pyg_graph = torch_geometric.utils.from_networkx(graph)
        pyg_graph.x = torch.zeros((pyg_graph.num_nodes,), dtype=torch.long)
        y_value = name_2_label[graph.name]
        pyg_graph.y = torch.full_like(pyg_graph.x, y_value)

        N = len(graph.nodes)
        graph.positions = {i: (i if i < N / 2 else N - i, 1 if i < N / 2 else -1) for i, x in enumerate(
            graph.nodes)}
        pyg_graph.graph = graph
        return pyg_graph

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def join_graphs(graphs):
    def merge_graphs(left_graph, right_graph):
        # 1) get end of left, start of right
        # 2) join with edge
        # 3) create new graph
        # 4) offset right graph positions
        return left_graph

    left_graph = graphs[0]
    for right_graph in graphs[1:]:
        left_graph = merge_graphs(left_graph, right_graph)
    return left_graph
