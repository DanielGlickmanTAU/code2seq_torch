from unittest import TestCase

import networkx as nx

import visualization
from graph_words import word_graphs


class Test(TestCase):
    def test_join_graphs(self):
        clique5 = word_graphs.clique_5
        cycle4 = word_graphs.cycle_4
        g = word_graphs.join_graphs([clique5, cycle4])
        visualization.draw_pyg_graph(g)
        self.assertEqual(len(g), len(cycle4) + len(clique5))
