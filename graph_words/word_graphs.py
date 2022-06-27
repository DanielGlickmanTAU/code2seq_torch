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
    if number == 33:
        return [lambda: Cycle(3)]
    if number == 0:
        return [lambda: Cycle(3), lambda: Cycle(4)]
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
                lambda: Dot(),
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
        self.dataset = WordsCombinationGraphDataset('global', graphs, len(graphs) * 10, words_per_row=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class WordsCombinationGraphDataset(Dataset):
    def set_color_if_all_nodes_have_same_shape(self, atoms_in_row, only_color):
        colors_in_rows = set(atom.nodes[0]['x'] for atom in atoms_in_row)
        shapes_in_rows = set(atom.nodes[0]['shape'] for atom in atoms_in_row)
        # col/row has same shape row and color
        if len(colors_in_rows) == 1 and (only_color or len(shapes_in_rows) == 1):
            for atom in atoms_in_row:
                for node in atom.nodes:
                    atom.nodes[node]['y'] = 1

    def __init__(self, color_mode, word_graphs, num_samples, words_per_row, num_rows=None, num_colors=2, edge_p=1.,
                 only_color=False, unique_atoms_per_example=False, unique_colors_per_example=False):
        if not num_rows:
            num_rows = words_per_row
        self.word_graphs = word_graphs
        self.name_2_label = {graph().name: i for i, graph in enumerate(word_graphs)}
        self.label_2_name = {i: graph().name for i, graph in enumerate(word_graphs)}
        self.dataset = []

        if color_mode == 'global':
            assert num_colors <= 1

        if color_mode == 'global':
            self.num_labels = len(self.name_2_label)
        elif color_mode == 'instance':
            self.num_labels = num_colors
        elif color_mode == 'both':
            self.num_labels = len(self.name_2_label) * num_colors
        # just 2 labels, match/ no match
        elif color_mode == 'rows':
            self.num_labels = 2
        else:
            raise Exception(f'unsupported color mode {color_mode}')

        for i in range(num_samples):
            if unique_atoms_per_example:
                num_unique_atoms = 2
                unique_atoms = numpy.random.choice(word_graphs, num_unique_atoms, replace=False)
                selected_words_ctors = numpy.random.choice(unique_atoms, words_per_row * num_rows)
            else:
                selected_words_ctors = numpy.random.choice(word_graphs, words_per_row * num_rows)

            if unique_colors_per_example:
                unique_colors = numpy.random.choice(list(range(1, num_colors + 1)), 2, replace=False)
                selected_colors = numpy.random.choice(unique_colors, words_per_row * num_rows)
            else:
                selected_colors = numpy.random.choice(list(range(1, num_colors + 1)), words_per_row * num_rows)
            selected_words = [g() for g in selected_words_ctors]

            # spit to rows
            words_in_grid = [selected_words[i:i + words_per_row] for i in
                             range(0, len(selected_words), words_per_row)]
            colors_in_grid = [selected_colors[i:i + words_per_row] for i in
                              range(0, len(selected_words), words_per_row)]
            if color_mode == 'instance' or color_mode == 'both':
                for row, color_row in zip(words_in_grid, colors_in_grid):
                    for word_instance, chosen_color in zip(row, color_row):
                        chosen_node = random.randint(0, len(word_instance) - 1)
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

            if color_mode == 'rows':
                # first color all nodes(x)
                for row, color_row in zip(words_in_grid, colors_in_grid):
                    for word_instance, chosen_color in zip(row, color_row):
                        for i, node in enumerate(word_instance.nodes):
                            word_instance.nodes[node]['shape'] = word_instance.name
                            word_instance.nodes[node]['y'] = 0
                            word_instance.nodes[node]['x'] = chosen_color

                # now go by rows and see if any contain all with same shape + color
                for i in range(len(words_in_grid)):
                    atoms_in_row = words_in_grid[i]
                    self.set_color_if_all_nodes_have_same_shape(atoms_in_row, only_color)
                    atoms_in_col = [words_in_grid[j][i] for j in range(len(words_in_grid))]
                    self.set_color_if_all_nodes_have_same_shape(atoms_in_col, only_color)

            if color_mode == 'global':
                for row in words_in_grid:
                    for word_instance in row:
                        for node in word_instance.nodes:
                            # word_instance.nodes[node]['color'] = word_instance.name
                            word_instance.nodes[node]['y'] = self.name_2_label[word_instance.name]
                            word_instance.nodes[node]['x'] = 0

            graph = join_graphs(words_in_grid, edge_p)
            # node_colors = [self.name_2_label[attr['color']] for _, attr in graph.nodes(data=True)]
            pyg_graph = create_pyg_graph(graph)
            self.dataset.append(pyg_graph)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_pyg_graph(graph):
    pyg_graph = torch_geometric.utils.from_networkx(graph)
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
