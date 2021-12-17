import collections
import concurrent.futures as futures
from typing import List

import networkx as nx
from tqdm import tqdm

from data.ast_conversion import config
from data.ast_conversion.efficent_impl import count_pairs_efficient, merge_nodes_efficient
from data.types import AST


def order_agnostic_name_merge(name1, name2):
    return f'{name1}@{name2}'


# mutates graphs
def learn_vocabulary(graphs: List[AST], vocab_size):
    def print_vocab(vocab):
        print('key, average frequency , average graph size')
        for r1, r2, freq, size in vocab:
            print(r1, r2, freq, size)

    vocab = []
    for i in tqdm(range(vocab_size)):
        # dict[str] -> list of (graph,parent index, child index)
        counter = collections.defaultdict(list)

        if config.parallel_compute:
            with futures.ThreadPoolExecutor() as executor:
                for i, graph in enumerate(graphs):
                    # executor.submit(count_pairs, graph, counter)
                    executor.submit(count_pairs_efficient, graph, counter, i)
        else:
            for i, graph in enumerate(graphs):
                count_pairs_efficient(graph, counter, i)

        sorted_counter = sorted(counter.items(), key=lambda item: -len(item[1]))
        best_key = sorted_counter[0][0]
        best_key_locations = sorted_counter[0][1]  # list of (graph,parent,child) tuples

        average_frequency = len(best_key_locations) / (len(graphs))
        average_graph_length = sum([len(g) for g in graphs]) / (len(graphs))
        best_key = best_key + (average_frequency, average_graph_length)

        vocab.append(best_key)

        merged_before = set()
        for i, (graphs_index, parent, child) in enumerate(best_key_locations):
            # if we have something like the rule a@a and a graph like (a -> a -> a),
            # we want to skip merging the 2ed and 3ed a(because the second is dead after merging it into the first).
            # can be removed if taking edge into account..
            if (graphs_index, parent) in merged_before:
                continue
            graph = graphs[graphs_index]
            # if graphs_index == 371 and parent == 183 and child == 186:
            #     assert 186 in graphs[372][183]['children']
            merge_nodes_efficient(graph, parent, child)
            # if graphs_index == 371 and parent == 183 and child == 186:
            #     assert 186 not in graphs[372][183]['children']
            # merge_nodes(graph, parent, child)
            merged_before.add((graphs_index, child))
            merged_before.add((graphs_index, parent))

            # for g_i, p_i, c_i in best_key_locations:
            #     if (g_i, p_i) not in merged_before and c_i not in graphs[g_i][p_i]['children']:
            #         print('!!!!')

    print_vocab(vocab)
    return [(r1, r2) for r1, r2, freq, size in vocab]


def count_pairs(g: nx.DiGraph, counter):
    for n, nbrs in g.adjacency():
        childs = nbrs.keys()
        parent_node = g.nodes[n]
        for child in childs:
            # consider ignoring childs with no children - terminals
            child_node = g.nodes[child]
            if config.skip_if_both_nodes_have_value and 'value' in parent_node and 'value' in child_node:
                pass
                # print(f'something strange.. value in both {g.nodes[n]} and {g.nodes[child]}')
            if not child_node['children']:
                pass
            else:
                counter[(parent_node['type'], child_node['type'])].append((g, n, child))


def merge_nodes(g: nx.DiGraph, parent: int, child: int):
    parent_node = g.nodes[parent]
    child_node = g.nodes[child]

    g.add_edges_from([parent, grand_children] for grand_children in list(g[child]))
    # if doing efficent counting,implement it about here....
    # todo check cycles
    parent_node['type'] = order_agnostic_name_merge(parent_node['type'], child_node['type'])
    g.remove_node(child)
