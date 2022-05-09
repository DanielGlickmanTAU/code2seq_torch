from code2seq.utils import compute

torch = compute.get_torch()
import itertools

import networkx as nx
import torch_geometric
from torch.utils.data import Dataset


class Clique:
    @staticmethod
    def _create_clique(n):
        G = nx.Graph()
        edges = itertools.combinations(range(0, n), 2)
        G.add_edges_from(edges)
        return G

    def __init__(self, n):
        self.n = n
        self.graph = self._create_clique(n)
        self.name = f'{n}_clique'


class Cycle:
    def __init__(self, n):
        self.n = n
        self.graph = nx.cycle_graph(n)
        self.name = f'{n}_cycle'


class WordGraphDataset(Dataset):
    def __init__(self):
        cycle_4 = Cycle(4)
        cycle_5 = Cycle(5)
        clique_4 = Clique(4)
        clique_5 = Clique(5)

        graphs = [cycle_4, cycle_5, clique_4, clique_5]

        self.name_2_label = {graph.name: i for i, graph in enumerate(graphs)}
        self.label_2_name = {i: graph.name for i, graph in enumerate(graphs)}

        self.dataset = []
        for graph in graphs:
            pyg_graph = self.create_pyg_graph(graph)
            self.dataset.append(pyg_graph)

    def create_pyg_graph(self, graph):
        pyg_graph = torch_geometric.utils.from_networkx(graph.graph)
        pyg_graph.x = torch.zeros((pyg_graph.num_nodes,), dtype=torch.long)
        y_value = self.name_2_label[graph.name]
        pyg_graph.y = torch.full_like(pyg_graph.x, y_value)

        N = len(graph.graph.nodes)
        graph.graph.positions = {i: (i if i < N / 2 else N - i, 1 if i < N/2 else -1) for i, x in enumerate(
            graph.graph.nodes)}
        pyg_graph.graph = graph.graph
        return pyg_graph

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
