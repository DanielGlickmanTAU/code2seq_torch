import itertools

import networkx as nx

import visualization
from graph_words import word_graphs

dataset = word_graphs.WordGraphDataset()

# draw graphs, each should have different color(by label)
for pyg_graph in dataset:
    visualization.draw(pyg_graph, pyg_graph.y, color_map={0: 'pink', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow'})
# color with x(no label) just as sanity
visualization.draw(pyg_graph, pyg_graph.x, color_map={0: 'gray'})

# join graphs

# join edges(last node to first node...)
