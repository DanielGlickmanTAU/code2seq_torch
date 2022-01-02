import collections
import concurrent.futures as futures
from typing import List

from tqdm import tqdm

from data import ast
from data.ast import AST
from data.ast_conversion import ast_to_graph
from data.node import draw_ast
import multiprocessing

vocab_separator = '@'


def order_agnostic_name_merge(name1, name2):
    return f'{name1}{vocab_separator}{name2}'


def should_count(g, node, child_node, max_word_joins, merging_2_value_nodes):
    both_have_type = 'type' in child_node and 'type' in node
    merged_less_than_max = both_have_type \
                           and node['type'].count(vocab_separator) + child_node['type'].count(
        vocab_separator) < max_word_joins
    if merging_2_value_nodes:
        return merged_less_than_max
    else:
        not_both_nodes_have_value = not ast.get_first_value(g, child_node) or not ast.get_first_value(g, node)
        return merged_less_than_max and not_both_nodes_have_value


# mutates graphs
def learn_vocabulary(graphs: List[AST], vocab_size, max_word_joins, scan_in_order=False, merging_2_value_nodes=True):
    if vocab_size == 0:
        return
    assert max_word_joins > 0

    for graph in graphs:
        ast_to_graph.add_parents(graph)

    counter_size_limit = 100_000

    def count_pairs_efficient(g, counter, i):
        for n in g:
            node = g[n]
            if 'children' not in node:
                continue
            for j in node['children']:
                child_node = g[j]
                if should_count(g, node, child_node, max_word_joins, merging_2_value_nodes):
                    node_type = node['type']
                    child_node_type = child_node['type']
                    types_id = (node_type, child_node_type)
                    locations_for_type = counter.get(types_id)
                    if locations_for_type:
                        locations_for_type.add((i, n, j))
                    elif len(counter) < counter_size_limit:
                        counter[types_id].add((i, n, j))

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

    # draw_graph_indexs = [0, 18, 22, 24, 31, 32, 33, 47]
    # for draw_graph_index in draw_graph_indexs:
    #     draw_ast(graphs[draw_graph_index], f'./images/fig_graph{draw_graph_index}_iter0')

    print('starting counting')
    # with futures.ProcessPoolExecutor() as executor:
    #     for i, graph in enumerate(graphs):
    #         executor.submit(count_pairs_efficient, graph, counter, i)
    for i, graph in enumerate(graphs):
        count_pairs_efficient(graph, counter, i)
    print('start merging')
    for i in tqdm(range(vocab_size)):
        # best_key_locations is  list of (graph,parent,child) tuples
        best_key, best_key_locations = max(counter.items(), key=lambda item: len(item[1]))

        average_frequency = len(best_key_locations) / (len(graphs))
        average_graph_length = sum([len(g) for g in graphs]) / (len(graphs))

        vocab.append(best_key)
        stats.append((average_frequency, average_graph_length))

        merge_locations(best_key_locations, counter, max_word_joins, merging_2_value_nodes, graphs)
        # for draw_graph_index in draw_graph_indexs:
        #     for g_i, index in merged_locations:
        #         if g_i == draw_graph_index:
        #             draw_ast(graphs[draw_graph_index], f'./images/fig_graph{draw_graph_index}_iter{i + 1}',
        #                      order_agnostic_name_merge(*best_key))
        #             break

    print_vocab([v + s for v, s in zip(vocab, stats)])
    return vocab


def merge_locations(best_key_locations, counter, max_word_joins, merging_2_value_nodes, graphs):
    merged_before = set()
    for graphs_index, parent, child in best_key_locations.copy():
        # if we have something like the rule a@a and a graph like (a -> a -> a),
        # we want to skip merging the 2ed and 3ed a(because the second is dead after merging it into the first).
        # can be removed if taking edge into account..
        if (graphs_index, parent) in merged_before:
            continue
        graph = graphs[graphs_index]
        merge_nodes_efficient(graph, parent, child, counter, graphs_index, max_word_joins, merging_2_value_nodes)

        merged_before.add((graphs_index, child))
        merged_before.add((graphs_index, parent))
    # return merged_before


def merge_nodes_efficient(g, parent: int, child: int, counter, graph_index, max_word_joins, merging_2_value_nodes):
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
            if should_count(g, parent_node, c, max_word_joins, merging_2_value_nodes):
                counter[(parent_node['type'], c['type'])].remove((graph_index, parent, c_i))

    def add_counts(parent, parent_node):
        for c_i in parent_node['children']:
            c = g[c_i]
            if should_count(g, parent_node, c, max_word_joins, merging_2_value_nodes):
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

    # when merging a,b in a->b->c.. need to change c's parents
    if 'children' in child_node:
        for child_child_i in child_node['children']:
            g[child_child_i]['parent'] = parent
    g.pop(child)
