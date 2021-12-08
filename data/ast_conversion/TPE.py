import collections
import concurrent.futures as futures
import networkx as nx
from tqdm import tqdm

from data.ast_conversion import config


def order_agnostic_name_merge(name1, name2):
    return f'{name1}@{name2}'


# mutates graphs
def learn_vocabulary(graphs, vocab_size):
    def print_vocab(vocab):
        print('key, average frequency , average graph size')
        for r1, r2, freq, size in vocab:
            print(r1, r2, freq, size)

    vocab = []
    for i in tqdm(range(vocab_size)):
        # dict[str] -> list of (graph,parent index, child index)
        counter = collections.defaultdict(list)

        with futures.ThreadPoolExecutor() as executor:
            for graph in graphs:
                executor.submit(count_pairs, graph, counter)
                # count_pairs_efficient(graph, counter)

        sorted_counter = sorted(counter.items(), key=lambda item: -len(item[1]))
        best_key = sorted_counter[0][0]
        best_key_locations = sorted_counter[0][1]  # list of (graph,parent,child) tuples

        average_frequency = len(best_key_locations) / (len(graphs))
        average_graph_length = sum([len(g) for g in graphs]) / (len(graphs))
        best_key = best_key + (average_frequency, average_graph_length)

        vocab.append(best_key)

        merged_before = set()
        for graph, parent, child in best_key_locations:
            # if we have something like the rule a@a and a graph like (a -> a -> a),
            # we want to skip merging the 2ed and 3ed a(because the second is dead after merging it into the first).
            if (graph, parent) in merged_before:
                continue
            # merge_nodes_efficient(graph, parent, child)
            merge_nodes(graph, parent, child)
            merged_before.add((graph, child))
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
