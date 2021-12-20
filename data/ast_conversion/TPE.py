import collections
import concurrent.futures as futures
from typing import List

from tqdm import tqdm

from data.types import AST

vocab_separator = '@'


def order_agnostic_name_merge(name1, name2):
    return f'{name1}{vocab_separator}{name2}'


# mutates graphs
def learn_vocabulary(graphs: List[AST], vocab_size, max_word_joins):
    assert max_word_joins > 0
    assert vocab_size > 0

    def count_pairs_efficient(g, counter, i):
        for n in g:
            node = g[n]
            if 'children' not in node:
                continue
            node_type = node['type']
            for j in node['children']:
                child_node = g[j]
                if 'type' in child_node:
                    child_node_type = child_node['type']
                    if node_type.count(vocab_separator) + child_node_type.count(vocab_separator) < max_word_joins:
                        counter[(node_type, child_node_type)].append((i, n, j))

    def merge_nodes_efficient(g, parent: int, child: int):
        def merge_children(parent_node, child_node):
            if child in parent_node['children']:
                new_children = [x for x in parent_node['children'] if x != child]
                new_children += child_node['children']
                parent_node['children'] = new_children
            else:
                print(f'bug! {parent_node}:{parent} , {child_node}:{child}')

        parent_node = g[parent]
        child_node = g[child]
        merge_children(parent_node, child_node)
        # if doing efficent counting,implement it about here....
        parent_node['type'] = order_agnostic_name_merge(parent_node['type'], child_node['type'])
        g.pop(child)

    def print_vocab(vocab):
        print('key, average frequency , average graph size')
        for r1, r2, freq, size in vocab:
            print(r1, r2, freq, size)

    vocab = []
    for i in tqdm(range(vocab_size)):
        # dict[str] -> list of (graph,parent index, child index)
        counter = collections.defaultdict(list)

        with futures.ThreadPoolExecutor() as executor:
            for i, graph in enumerate(graphs):
                executor.submit(count_pairs_efficient, graph, counter, i)

        sorted_counter = sorted(counter.items(), key=lambda item: -len(item[1]))
        best_key = sorted_counter[0][0]
        best_key_locations = sorted_counter[0][1]  # list of (graph,parent,child) tuples

        average_frequency = len(best_key_locations) / (len(graphs))
        average_graph_length = sum([len(g) for g in graphs]) / (len(graphs))
        best_key = best_key + (average_frequency, average_graph_length)

        vocab.append(best_key)

        merged_before = set()
        for graphs_index, parent, child in best_key_locations:
            # if we have something like the rule a@a and a graph like (a -> a -> a),
            # we want to skip merging the 2ed and 3ed a(because the second is dead after merging it into the first).
            # can be removed if taking edge into account..
            if (graphs_index, parent) in merged_before:
                continue
            graph = graphs[graphs_index]
            merge_nodes_efficient(graph, parent, child)

            merged_before.add((graphs_index, child))
            merged_before.add((graphs_index, parent))

    print_vocab(vocab)
    return [(r1, r2) for r1, r2, freq, size in vocab]
