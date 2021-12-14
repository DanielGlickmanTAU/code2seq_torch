import argparse
import multiprocessing
from pathlib import Path
import os

import numpy as np

from data.ast_conversion import ast_to_graph
from data.ast_conversion.ast_to_graph import collect_asts
import TPE

parser = argparse.ArgumentParser()
data_dir = os.getcwd().split('\data')[0] + '\data\python'
parser.add_argument('--data_dir', default=data_dir, type=str)
parser.add_argument('--valid_p', type=float, default=0.2)
parser.add_argument('--max_path_length', type=int, default=8)
parser.add_argument('--max_path_width', type=int, default=2)
parser.add_argument('--use_method_name', type=bool, default=True)
parser.add_argument('--use_nums', type=bool, default=True)
parser.add_argument('--output_dir', default='out_python', type=str)
parser.add_argument('--n_jobs', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--seed', type=int, default=239)


args = parser.parse_args()
np.random.seed(args.seed)

data_dir = Path(args.data_dir)
limit = 10
# limit = 0
evals = collect_asts(data_dir / 'python50k_eval.json', limit=limit)
train = collect_asts(data_dir / 'python100k_train.json', limit=limit)

graphs_eval = ast_to_graph.collect_all_ast_graphs(evals, args)
graphs_train = ast_to_graph.collect_all_ast_graphs(train, args)
# graphs = collect_all_ast_graphs(evals, args, collection_function=ast_to_graph.filter_ast)

vocab_size = 10

vocab = TPE.learn_vocabulary(graphs_eval, vocab_size)
data = [ast_to_graph.graph_to_ast(graph) for graph in graphs_eval]

# replace [ [] , []] with [] \n []
ast_to_graph.write_asts_to_file(data_dir / 'python50k_eval_formatted.json', data)

# data = [ast_to_graph.graph_to_ast(graph) for graph in graphs_train]
# ast_to_graph.write_asts_to_file(data_dir / 'python100k_train_formatted.json', data)

# with open(output_file, 'w') as f:
#     for line_index, line in enumerate(samples):
#         f.write(line + ('' if line_index == len(samples) - 1 else '\n'))

# output_dir = Path(args.output_dir)
# output_dir.mkdir(exist_ok=True)
# for split_name, split in zip(
#
#         ('test',),
#         (evals,)
# ):
#     output_file = output_dir / f'{split_name}_output_file.txt'
#     __collect_all_and_save(split, args, output_file)
