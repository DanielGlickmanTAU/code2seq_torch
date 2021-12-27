import itertools
import json

import joblib
import networkx as nx
import tqdm
from typing import List, Dict

from data.ast import AST
import multiprocessing

def create_nx_graph(ast):
    g = nx.convert.from_dict_of_dicts(
        {id: {child_id: [] for child_id in node['children']} for id, node in ast.items()},
        create_using=nx.DiGraph
    )
    # g.add_edges_from(edges)

    for node in g:
        if 'type' in ast[node]:
            g.nodes[node]['type'] = ast[node]['type']
        if 'value' in ast[node]:
            g.nodes[node]['value'] = ast[node]['value']

    return g


def extract_function_subtree(ast: AST, root_index) -> AST:
    root = ast[root_index]
    assert root['type'].startswith('FunctionDef')

    new_ast = {}
    to_visit = [root_index]
    while len(to_visit) > 0:
        index = to_visit.pop()
        node_dict = ast[index].copy()
        if 'children' not in node_dict:
            node_dict['children'] = []
        to_visit += node_dict['children']
        new_ast[index] = node_dict

    return new_ast


def graph_to_ast(graph):
    fixer = {original_index: fixed_index for (fixed_index, original_index) in enumerate(list(graph))}
    data = []
    for n in graph:
        node_as_dict = graph.nodes[n]
        node_as_dict['children'] = [fixer[original_child_index] for original_child_index in graph[n]]
        data.append(node_as_dict)

    return data


def write_asts_to_file(path, asts: List[AST]):
    for ast in asts:
        assert_all_nodes_contain_either_type_or_value_and_all_children_are_in_dict(ast)

    with open(path, 'w') as f:
        for line in asts:
            json.dump(line, f)
            f.write('\n')


def assert_all_nodes_contain_either_type_or_value_and_all_children_are_in_dict(ast: AST):
    def assert_all_children_in_ast(node):
        if 'children' in node:
            for child in node['children']:
                assert child in ast

    for node in ast.values():
        assert ("value" in node) ^ ('type' in node), f'node {node} must contain either value or node.'
        assert_all_children_in_ast(node)


def collect_asts(json_file, limit=0) -> List[AST]:
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

            if isinstance(ast, List):
                ast = convert(ast)

            # convert keys to int from string
            ast = with_int_keys(ast)

            assert_all_nodes_contain_either_type_or_value_and_all_children_are_in_dict(ast)
            asts.append(ast)
            if len(asts) % 10_000 == 0:
                print(f'done {len(asts)}')
            if limit and len(asts) > limit:
                break

    return asts


def with_int_keys(ast):
    return {int(key): value for key, value in ast.items()}


def add_parents(ast: AST):
    for node_id, node in ast.items():
        if 'children' in node:
            for child in node['children']:
                ast[child]['parent'] = node_id


def convert(ast) -> AST:
    increase_by = {}  # count of how many idx to increase the new idx by:
    # each time there is a value node
    cur = 0

    for i, node in enumerate(ast):
        increase_by[i] = cur
        if "value" in node and 'type' in node:
            cur += 1

    if cur == 0:
        return {i: node for i, node in enumerate(ast)}

    new_dp = []
    for i, node in enumerate(ast):
        inc = increase_by[i]
        if "value" in node:
            child = [i + inc + 1]
            if "children" in node:
                child += [n + increase_by[n] for n in node["children"]]
            new_dp.append({"type": node["type"], "children": child})
            new_dp.append({"value": node["value"]})
        else:
            if "children" in node:
                node["children"] = [n + increase_by[n] for n in node["children"]]
            else:
                node["children"] = []
            new_dp.append(node)

    # sanity check
    children = []
    for node in new_dp:
        if "children" in node:
            children += node["children"]
    assert len(children) == len(set(children))
    return {i: node for i, node in enumerate(new_dp)}


def __collect_ast_graphs(ast, collection_function=extract_function_subtree):
    samples = []
    for node_index, node in ast.items():
        if 'type' in node and 'FunctionDef' in node['type']:
            sample = collection_function(ast, node_index)
            if sample is not None:
                samples.append(sample)
    return samples


def collect_all_ast_graphs(asts, args, collection_function=extract_function_subtree) -> List[AST]:
    # pool = multiprocessing.Pool()


    parallel = joblib.Parallel(n_jobs=args.n_jobs)
    func = joblib.delayed(lambda ast: __collect_ast_graphs(ast, collection_function=collection_function))

    samples = parallel(func(ast) for ast in asts)
    return list(itertools.chain.from_iterable(samples))


def collect_all_functions(path, args, limit=0):
    asts = collect_asts(path, limit=limit)
    return collect_all_ast_graphs(asts, args)
