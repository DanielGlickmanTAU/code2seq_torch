import argparse
import multiprocessing
from pathlib import Path
import os
from typing import cast

import json
from omegaconf import DictConfig, OmegaConf
import networkx as nx
# from code2seq.data.path_context_data_module import PathContextDataModule
from data.ast_conversion import ast_to_graph, TPE
from data.ast_conversion.ast_to_graph import __collect_asts
import data.py150k_extractor as py_extractor

parser = argparse.ArgumentParser()
data_dir = os.getcwd().split('/data')[0] + '/data/python'
parser.add_argument('--data_dir', default=data_dir, type=str)
parser.add_argument('--valid_p', type=float, default=0.2)
parser.add_argument('--max_path_length', type=int, default=8)
parser.add_argument('--max_path_width', type=int, default=2)
parser.add_argument('--use_method_name', type=bool, default=True)
parser.add_argument('--use_nums', type=bool, default=True)
parser.add_argument('--output_dir', default='out_python', type=str)
parser.add_argument('--n_jobs', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--seed', type=int, default=239)
parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str,
                    default=os.getcwd().split('/code2seq_torch')[0] + '/code2seq_torch/config/code2seq-py150k.yaml')

args = parser.parse_args()
data_dir = Path(args.data_dir)


limit = 0
vocab_size = 20
# limit = 0
evals = ast_to_graph.__collect_asts(data_dir / 'python50k_eval.json', limit=limit)
trains = ast_to_graph.__collect_asts(data_dir / 'python100k_train.json', limit=limit)

graphs_eval = ast_to_graph.__collect_all_ast_graphs(evals, args)
graphs_train = ast_to_graph.__collect_all_ast_graphs(trains, args)


#notice graphs_eval + graphs_train here
vocab = TPE.learn_vocabulary(graphs_eval+graphs_train, vocab_size)
data_eval = [ast_to_graph.graph_to_ast(graph) for graph in graphs_eval]
data_train = [ast_to_graph.graph_to_ast(graph) for graph in graphs_train]

eval_compressed_graphs_file = data_dir / f'python50k_eval_compressed_{len(vocab)}.json'
train_compressed_graphs_file = data_dir / f'python100k_train_compressed_{len(vocab)}.json'
ast_to_graph.write_asts_to_file(eval_compressed_graphs_file, data_eval)
ast_to_graph.write_asts_to_file(train_compressed_graphs_file, data_train)

vocab_file = data_dir / f'vocab_{len(vocab)}'
json.dump(vocab, open(vocab_file, 'w'))
