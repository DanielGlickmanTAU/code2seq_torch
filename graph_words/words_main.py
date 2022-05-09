import itertools

import networkx as nx

import visualization
from graph_words import word_graphs

cycle_4 = word_graphs.Cycle(4)
cycle_5 = word_graphs.Cycle(5)
clique_4 = word_graphs.Clique(4)
clique_5 = word_graphs.Clique(5)

graphs = [cycle_4, cycle_5, clique_4, clique_5]

# draw graphs
for graph in graphs:
    visualization.draw_pyg_graph(graph.graph)

# color graphs

# join graphs

# join edges(last node to first node...)
