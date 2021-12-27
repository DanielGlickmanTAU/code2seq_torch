import argparse
import gc
import os
import re
import itertools
from pathlib import Path
from typing import List

import tqdm
import joblib
import numpy as np
import sklearn.model_selection as model_selection

from data.ast import get_first_value, get_node_values
from data.ast_conversion.ast_to_graph import collect_asts

token_separator = '|'

METHOD_NAME, NUM = 'METHODNAME', 'NUM'


# returns all the paths from root to terminals. as list of lists
# e.g 1,2,3,num
def __terminals(ast, node_index, args):
    stack, paths = [], []

    def dfs(v):
        stack.append(v)

        v_node = ast[v]

        values = get_node_values(ast, v_node)
        if values:
            if v == node_index:  # Top-level func def node.
                if args.use_method_name:
                    paths.append((stack.copy(), METHOD_NAME))
            # else:
            v_type = v_node['type']

            if 'Name' in v_type:
                # paths.append((stack.copy(), v_node['value']))
                for value in values:
                    paths.append((stack.copy(), value))
            # elif args.use_nums and 'Num' in v_type:
            if args.use_nums and 'Num' in v_type:
                paths.append((stack.copy(), NUM))

        if 'children' in v_node:
            for child in v_node['children']:
                dfs(child)

        stack.pop()

    dfs(node_index)

    return paths


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def __raw_tree_paths(ast, node_index, args):
    # list of ([list path form root to terminal], terminal name)
    # includes root and includes AttributeLoad parent
    tnodes = __terminals(ast, node_index, args)

    tree_paths = []
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
            iterable=tnodes,
            r=2,
    ):
        # lca is the node in which the paths meet. prefix is v to lca. prefix is from lca to u
        prefix, lca, suffix = __merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= args.max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= args.max_path_width):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)
        else:
            pass

    return tree_paths


# ArrayList -> array|list
def __delim_name(name):
    if name in {METHOD_NAME, NUM}:
        return name

    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))

    return token_separator.join(block.lower() for block in blocks)


# returns strings
def _collect_sample(ast, fd_index, args):
    root = ast[fd_index]
    if not root['type'].startswith('FunctionDef'):
        raise ValueError('Wrong node type.')

    # target = root['value']
    target = get_first_value(ast, root)
    assert target is not None

    # tree_paths format is (target,list of node ids, source)
    tree_paths = __raw_tree_paths(ast, fd_index, args)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        if finish == target or start == target:
            continue

        start, finish = __delim_name(start), __delim_name(finish)
        in_connector = (ast[v]['type'] for v in connector)
        connector = '|'.join(in_connector)

        context = f'{start},{connector},{finish}'
        contexts.append(context)

    if len(contexts) == 0:
        return None

    target = __delim_name(target)
    context = ' '.join(contexts)

    return f'{target} {context}'


# returns list of strings
def __collect_samples(ast, args):
    samples = []
    for node_index, node in ast.items():
        if 'type' in node and node['type'].startswith('FunctionDef'):
            # can be called more than once. if 2 methods in a file.
            sample = _collect_sample(ast, node_index, args)
            if sample:
                samples.append(sample)

    return samples


def collect_all_and_save(asts, args, output_file, para=True):
    samples = collect_all(asts, args, para)

    write_to_file(output_file, samples)


def write_to_file(output_file: str, samples: List):
    with open(output_file, 'w') as f:
        for line_index, line in enumerate(samples):
            f.write(line + ('' if line_index == len(samples) - 1 else '\n'))


def collect_all(asts, args, para):
    parallel = joblib.Parallel(args.n_jobs)

    func = joblib.delayed(__collect_samples)
    if para:
        samples = parallel(func(ast, args) for ast in tqdm.tqdm(asts))
        samples = list(itertools.chain.from_iterable(samples))
    else:
        samples = [__collect_samples(ast, args) for ast in tqdm.tqdm(asts)]
        samples = list(itertools.chain.from_iterable(samples))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='python', type=str)
    parser.add_argument('--valid_p', type=float, default=0.2)
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)
    parser.add_argument('--use_method_name', type=bool, default=True)
    parser.add_argument('--use_nums', type=bool, default=True)
    parser.add_argument('--output_dir', default='out_python', type=str)
    parser.add_argument('--n_jobs', type=int, default=min(multiprocessing.cpu_count(), 4))
    parser.add_argument('--seed', type=int, default=239)

    args = parser.parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    # limit = 100
    limit = 0

    compressed_vocab_size = 'compressed_50'
    trains = collect_asts(data_dir / ('python100k_train_%s.json' % compressed_vocab_size), limit=limit)
    train, valid = model_selection.train_test_split(
        trains,
        test_size=args.valid_p,
    )

    evals = collect_asts(data_dir / ('python50k_eval_%s.json' % compressed_vocab_size), limit=limit)
    test = evals

    output_dir = Path(args.output_dir + '/' + compressed_vocab_size)
    output_dir.mkdir(exist_ok=True)
    out_files = []
    for split_name, split in zip(
            ('train', 'val', 'test'),
            # ('test',),
            (train, valid, test),
            # (test,),

    ):
        output_file = output_dir / f'{split_name}.c2s'
        collect_all_and_save(split, args, output_file, para=True)
        del split
        gc.collect()
        out_files.append(str(out_files))

    os.system(f'tar cvzf  python_{compressed_vocab_size}_c2s.tar.gz {" ".join(out_files)}')


if __name__ == '__main__':
    main()
