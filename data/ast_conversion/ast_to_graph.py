import itertools
import json

import joblib
import networkx as nx
import tqdm


def create_graph(ast, root_index):
    root = ast[root_index]
    assert root['type'].startswith('FunctionDef')

    edges = []
    # value not interesting... only to restore.. need node['type']
    to_visit = [root_index]
    while len(to_visit) > 0:
        index = to_visit.pop()
        node_dict = ast[index]
        if 'children' in node_dict:
            edges += [(index, child) for child in node_dict['children']]
            to_visit += node_dict['children']
    g = nx.DiGraph()
    g.add_edges_from(edges)

    for node in g:
        g.nodes[node]['type'] = ast[node]['type']
        if 'value' in ast[node]:
            g.nodes[node]['value'] = ast[node]['value']

    return g


def filter_ast(ast, root_index):
    root = ast[root_index]
    assert root['type'].startswith('FunctionDef')

    new_ast = {}
    to_visit = [root_index]
    while len(to_visit) > 0:
        index = to_visit.pop()
        node_dict = ast[index].copy()
        # new_ast.append(node_dict)
        new_ast[index] = node_dict
        if 'children' not in node_dict:
            node_dict['children'] = []
        # visit by old index
        to_visit += node_dict['children']
        # parent_index = len(new_ast)
        # node_dict['children'] = [parent_index + child_num for child_num in range(len(node_dict['children']))]
        # node_dict['children'] = [ast[i] for i in node_dict['children']]

    return new_ast


def graph_to_ast(graph):
    fixer = {original_index: fixed_index for (fixed_index, original_index) in enumerate(list(graph))}
    data = []
    for n in graph:
        node_as_dict = graph.nodes[n]
        node_as_dict['children'] = [fixer[original_child_index] for original_child_index in graph[n]]
        data.append(node_as_dict)

    return data


def write_asts_to_file(path, asts):
    with open(path, 'w') as f:
        for line in asts:
            json.dump(line, f)
            f.write('\n')


def __collect_asts(json_file, limit=0):
    asts = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            try:
                ast = json.loads(line.strip())
            except:
                print('warning! failed parsing json')
                continue
            if len(ast) == 0:
                continue
            asts.append(ast)
            if len(asts) % 10_000 == 0:
                print(f'done {len(asts)}')
            if limit and len(asts) > limit:
                break

    return asts


def __collect_ast_graphs(ast, args=None, collection_function=create_graph):
    samples = []
    for node_index, node in enumerate(ast):
        if node['type'].startswith('FunctionDef'):
            sample = collection_function(ast, node_index)
            if sample is not None:
                samples.append(sample)
    return samples


def __collect_all_ast_graphs(asts, args, collection_function=create_graph):
    parallel = joblib.Parallel(n_jobs=args.n_jobs)
    func = joblib.delayed(lambda ast, args: __collect_ast_graphs(ast, collection_function=collection_function))

    samples = parallel(func(ast, args) for ast in asts)
    return list(itertools.chain.from_iterable(samples))
