import itertools

import networkx as nx


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
