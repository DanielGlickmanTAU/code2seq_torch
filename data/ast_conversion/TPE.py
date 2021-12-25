import collections
import concurrent.futures as futures
from typing import List

from tqdm import tqdm

from data import ast
from data.ast import AST
from data.ast_conversion import ast_to_graph
from data.node import draw_ast

vocab_separator = '@'


def order_agnostic_name_merge(name1, name2):
    return f'{name1}{vocab_separator}{name2}'


def should_count(g, node, child_node, max_word_joins, from_bottom=True):
    both_have_type = 'type' in child_node and 'type' in node
    merged_less_than_max = both_have_type \
                           and node['type'].count(vocab_separator) + child_node['type'].count(
        vocab_separator) < max_word_joins
    only_child_have_value = not from_bottom or (ast.get_first_value(g, child_node) and not ast.get_first_value(g, node))
    return merged_less_than_max and only_child_have_value


# mutates graphs
def learn_vocabulary(graphs: List[AST], vocab_size, max_word_joins, scan_in_order=False, from_bottom=True):
    assert max_word_joins > 0
    assert vocab_size > 0
    for graph in graphs:
        ast_to_graph.add_parents(graph)

    def count_pairs_efficient(g, counter, i):
        for n in g:
            node = g[n]
            if 'children' not in node:
                continue
            for j in node['children']:
                child_node = g[j]
                if should_count(g, node, child_node, max_word_joins, from_bottom):
                    node_type = node['type']
                    child_node_type = child_node['type']
                    counter[(node_type, child_node_type)].add((i, n, j))
                    if scan_in_order:
                        break

    def count_all_children(g, counter, i):
        for n in g:
            node = g[n]
            if 'children' not in node:
                continue
            if 'type' not in node:
                continue
            node_type = node['type']
            joined_types = [node_type]
            # todo use should_count here
            for j in node['children']:
                child_node = g[j]
                if 'type' in child_node:
                    child_node_type = child_node['type']
                    if node_type.count(vocab_separator) + child_node_type.count(vocab_separator) < max_word_joins:
                        joined_types.append(child_node_type)
            if len(joined_types) > 1:
                counter['@'.join(joined_types)].add((i, n))

    def print_vocab(vocab):
        print('key, average frequency , average graph size')
        for r1, r2, freq, size in vocab:
            print(r1, r2, freq, size)

    vocab = []
    stats = []
    # dict[str] -> list of (graph,parent index, child index)
    counter = collections.defaultdict(set)

    with futures.ThreadPoolExecutor() as executor:
        for i, graph in enumerate(graphs):
            executor.submit(count_pairs_efficient, graph, counter, i)

    # draw_ast(graphs[0])

    for i in tqdm(range(vocab_size)):
        sorted_counter = sorted(counter.items(), key=lambda item: -len(item[1]))
        best_key = sorted_counter[0][0]
        best_key_locations = sorted_counter[0][1]  # list of (graph,parent,child) tuples

        average_frequency = len(best_key_locations) / (len(graphs))
        average_graph_length = sum([len(g) for g in graphs]) / (len(graphs))

        vocab.append(best_key)
        stats.append((average_frequency, average_graph_length))

        merge_locations(best_key_locations, counter, max_word_joins, from_bottom, graphs)

    print_vocab([v + s for v, s in zip(vocab, stats)])
    return vocab


def merge_locations(best_key_locations, counter, max_word_joins, from_bottom, graphs):
    merged_before = set()
    for graphs_index, parent, child in best_key_locations.copy():
        # if we have something like the rule a@a and a graph like (a -> a -> a),
        # we want to skip merging the 2ed and 3ed a(because the second is dead after merging it into the first).
        # can be removed if taking edge into account..
        if (graphs_index, parent) in merged_before:
            continue
        graph = graphs[graphs_index]
        merge_nodes_efficient(graph, parent, child, counter, graphs_index, max_word_joins, from_bottom)

        merged_before.add((graphs_index, child))
        merged_before.add((graphs_index, parent))


def merge_nodes_efficient(g, parent: int, child: int, counter, graph_index, max_word_joins, from_bottom):
    def merge_children(parent_node, child_node):
        if child in parent_node['children']:
            new_children = [x for x in parent_node['children'] if x != child] + child_node['children']
            parent_node['children'] = new_children
            parent_node['type'] = order_agnostic_name_merge(parent_node['type'], child_node['type'])
        else:
            print(f'bug! {parent_node}:{parent} , {child_node}:{child}')

    def remove_counts(parent, parent_node):
        if 'children' not in parent_node:
            return
        for c_i in parent_node['children']:
            c = g[c_i]
            if should_count(g, parent_node, c, max_word_joins, from_bottom):
                counter[(parent_node['type'], c['type'])].remove((graph_index, parent, c_i))

    def add_counts(parent, parent_node):
        for c_i in parent_node['children']:
            c = g[c_i]
            if should_count(g, parent_node, c, max_word_joins, from_bottom):
                counter[(parent_node['type'], c['type'])].add((graph_index, parent, c_i))

    parent_node = g[parent]
    child_node = g[child]

    parent_parent = parent_node['parent'] if 'parent' in parent_node else None
    parent_parent_node = g[parent_parent] if parent_parent is not None else None

    remove_counts(parent, parent_node)
    remove_counts(child, child_node)
    if parent_parent is not None:
        remove_counts(parent_parent, parent_parent_node)

    merge_children(parent_node, child_node)
    add_counts(parent, parent_node)
    if parent_parent is not None:
        add_counts(parent_parent, parent_parent_node)

    if 'children' in child_node:
        for child_child_i in child_node['children']:
            g[child_child_i]['parent'] = parent
    g.pop(child)
