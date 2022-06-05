import math
import random

import numpy
from typing import List

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
    pos = circle_sections(n)
    graph.positions = {i: pos for i, pos in zip(graph, pos)}
    return graph


def Cycle(n):
    graph = nx.cycle_graph(n)
    graph.name = f'{n}_cycle'
    pos = circle_sections(n)
    graph.positions = {i: pos for i, pos in zip(graph, pos)}
    return graph


def JoinedCycles():
    graph = nx.cycle_graph(10)
    graph.add_edges_from([(0, 5)])
    graph.name = f'Jcycle'
    pos = circle_sections(10)
    graph.positions = {i: pos for i, pos in zip(graph, pos)}
    return graph


def ChordCycle(size=6):
    graph = nx.cycle_graph(size)
    graph.add_edges_from([(0, int(size / 2))])
    graph.name = f'ChordCycle_{size}'
    pos = circle_sections(size)
    graph.positions = {i: pos for i, pos in zip(graph, pos)}
    return graph


def Tree_small():
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2)]
    )
    graph.name = f'tree_small'
    graph.positions = {0: (0, 1), 1: (0.5, 0), 2: (-0.5, 0)}
    return graph


def Tree_large():
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2),
         (1, 3), (1, 4),
         (2, 5), (2, 6)
         ]

    )
    graph.name = f'tree_large'
    graph.positions = {0: (0, 1), 1: (-0.5, 0), 2: (0.5, 0),
                       3: (-0.75, -0.7), 4: (-0.25, -0.7),
                       5: (0.25, -0.7), 6: (0.75, -0.7)
                       }
    return graph


def Dot():
    graph = nx.Graph()
    graph.add_node(0)
    graph.name = f'dot'
    graph.positions = {0: (0, 0)
                       }
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


def JoinedSquared():
    graph = nx.Graph()
    graph.name = 'jsquare'
    graph.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 0),
         (1, 4), (4, 5), (5, 2)]
    )
    return graph


def get_atom_set(number):
    if number == 1:
        return [lambda: Cycle(4), lambda: Cycle(5), lambda: Cycle(6), lambda: Tree_large()]
    if number == 2:
        return [lambda: Cycle(5), lambda: Cycle(6),
                lambda: JoinedCycles(), lambda: Tree_large()]
    if number == 3:
        return [lambda: Cycle(5), lambda: Cycle(6), lambda: Tree_small(),
                lambda: JoinedCycles(), lambda: Tree_large()]
    if number == 4:
        return [lambda: Cycle(5), lambda: Cycle(6), lambda: Tree_small(),
                lambda: JoinedCycles(), lambda: Tree_large(), lambda: ChordCycle()]
    if number == 5:
        return [lambda: Cycle(3), lambda: Cycle(4), lambda: Cycle(5), lambda: Cycle(6), lambda: Tree_small(),
                lambda: JoinedCycles(), lambda: Tree_large(), lambda: ChordCycle()]
    if number == 6:
        return [lambda: Cycle(3), lambda: Cycle(4), lambda: Cycle(5), lambda: Cycle(6), lambda: Tree_small(),
                lambda: JoinedCycles(), lambda: Tree_large(), lambda: ChordCycle()]

    if number == 7:
        return [lambda: Cycle(4), lambda: Cycle(5), lambda: Cycle(6),
                lambda: Tree_small(), lambda: Tree_large(),
                # lambda: Dot(),
                lambda: JoinedCycles(), lambda: ChordCycle(5), lambda: ChordCycle(6)]

    raise Exception(f'unknown atom set option {number}')


def circle_sections(divisions, radius=1):
    # the difference between angles in radians -- don't bother with degrees
    angle = 2 * math.pi / divisions

    # a list of all angles using a list comprehension
    angles = [i * angle for i in range(divisions)]

    # finally return the coordinates on the circle as a list of 2-tuples
    return [(radius * math.cos(a), radius * math.sin(a)) for a in angles]


class WordGraphDataset(Dataset):
    def __init__(self, graphs):
        self.dataset = WordsCombinationGraphDataset('global', graphs, len(graphs) * 10, words_per_sample=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class WordsCombinationGraphDataset(Dataset):
    def __init__(self, color_mode, word_graphs, num_samples, words_per_sample, num_rows=1, num_colors=2, edge_p=1.):
        self.word_graphs = word_graphs
        self.num_labels = num_colors
        self.name_2_label = {graph().name: i for i, graph in enumerate(word_graphs)}
        self.label_2_name = {i: graph().name for i, graph in enumerate(word_graphs)}
        self.dataset = []

        if color_mode == 'global':
            self.num_labels = len(self.name_2_label)
        elif color_mode == 'instance':
            self.num_labels = num_colors
        elif color_mode == 'both':
            self.num_labels = len(self.name_2_label) * num_colors
        else:
            raise Exception(f'unsupported color mode {color_mode}')

        for i in range(num_samples):
            selected_words_ctors = numpy.random.choice(word_graphs, words_per_sample * num_rows)
            selected_words = [g() for g in selected_words_ctors]

            # spit to rows
            words_in_grid = [selected_words[i:i + words_per_sample] for i in
                             range(0, len(selected_words), words_per_sample)]
            if color_mode == 'instance' or color_mode == 'both':
                for row in words_in_grid:
                    for word_instance in row:
                        chosen_node = random.randint(0, len(word_instance) - 1)
                        chosen_color = random.randint(1, num_colors)
                        for i, node in enumerate(word_instance.nodes):
                            word_instance.nodes[node]['shape'] = word_instance.name
                            if color_mode == 'instance':
                                # -1 is so it will play nice with drawing the right color..
                                word_instance.nodes[node]['y'] = chosen_color - 1
                            elif color_mode == 'both':
                                word_instance.nodes[node]['y'] = self.name_2_label[
                                                                     word_instance.name] * num_colors + chosen_color - 1

                            if i == chosen_node:
                                word_instance.nodes[node]['x'] = chosen_color
                            else:
                                word_instance.nodes[node]['x'] = 0

            if color_mode == 'global':
                for row in words_in_grid:
                    for word_instance in row:
                        for node in word_instance.nodes:
                            # word_instance.nodes[node]['color'] = word_instance.name
                            word_instance.nodes[node]['y'] = self.name_2_label[word_instance.name]
                            word_instance.nodes[node]['x'] = 0

            graph = join_graphs(words_in_grid, edge_p)
            # node_colors = [self.name_2_label[attr['color']] for _, attr in graph.nodes(data=True)]
            node_colors = None
            pyg_graph = create_pyg_graph(graph, node_colors)
            self.dataset.append(pyg_graph)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_pyg_graph(graph, node_colors: List[int]):
    pyg_graph = torch_geometric.utils.from_networkx(graph)
    # pyg_graph.x = torch.zeros((pyg_graph.num_nodes,), dtype=torch.long)

    # pyg_graph.y = torch.tensor(node_colors)

    if not hasattr(graph, 'positions'):
        graph.positions = None
    pyg_graph.graph = graph
    return pyg_graph


def join_graphs_old(graphs):
    def merge_graphs(left_graph, right_graph):
        new_graph = nx.disjoint_union(left_graph, right_graph)
        left_graph_end_edge, right_graph_start_edge = len(left_graph) - 1, len(left_graph)
        new_graph.add_edge(left_graph_end_edge, right_graph_start_edge)

        return new_graph

    left_graph = graphs[0]
    for right_graph in graphs[1:]:
        left_graph = merge_graphs(left_graph, right_graph)
    return left_graph


def join_graphs(graphs, edge_p=1.):
    def select_random_node(graph_index):
        graph_lowest_node_id = first_labels[graph_index]
        return numpy.random.randint(graph_lowest_node_id, graph_lowest_node_id + len(flat_graphs[graph_index]))

    assert isinstance(graphs[0], list), f' expects list of list(grid) not {graphs}'

    first_labels = [0]

    flat_graphs = [graph for row in graphs for graph in row]

    for G in flat_graphs[:-1]:
        first_labels.append(len(G) + first_labels[-1])

    relabeled = [
        nx.convert_node_labels_to_integers(G, first_label=first_label)
        for G, first_label in zip(flat_graphs, first_labels)
    ]
    R = nx.union_all(relabeled)

    positions = []
    for i_row, row in enumerate(graphs):
        for i, G in enumerate(row):
            R.graph.update(G.graph)
            for (x, y) in G.positions.values():
                # shift x position
                positions.append((x + 3 * i, y - 3 * i_row))

    assert len(R) == len(positions)
    R.positions = {node: pos for node, pos in zip(R, positions)}

    # connect to right in same row
    for i, row in enumerate(graphs):
        for j, left_graph in enumerate(row[:-1]):
            if random.random() > edge_p:
                continue
            graph_num = i * len(row) + j
            left_node = select_random_node(graph_num)
            right_node = select_random_node(graph_num + 1)
            R.add_edge(left_node, right_node)

    # connect down, with row below
    for i, row in enumerate(graphs[:-1]):
        for j, left_graph in enumerate(row):
            graph_num = i * len(row) + j
            left_node = select_random_node(graph_num)
            right_node = select_random_node(graph_num + len(row))
            R.add_edge(left_node, right_node)

    return R
