import argparse
import multiprocessing
from pathlib import Path
import os
from typing import cast

from omegaconf import DictConfig, OmegaConf
import networkx as nx
from code2seq.data.path_context_data_module import PathContextDataModule
from data.ast_conversion import ast_to_graph, TPE
from data.ast_conversion.ast_to_graph import __collect_asts
import data.py150k_extractor as py_extractor

parser = argparse.ArgumentParser()
data_dir = os.getcwd().split('\data')[0] + '\data\python'
parser.add_argument('--data_dir', default=data_dir, type=str)
parser.add_argument('--valid_p', type=float, default=0.2)
parser.add_argument('--max_path_length', type=int, default=999)
parser.add_argument('--max_path_width', type=int, default=999)
parser.add_argument('--use_method_name', type=bool, default=True)
parser.add_argument('--use_nums', type=bool, default=True)
parser.add_argument('--output_dir', default='out_python', type=str)
parser.add_argument('--n_jobs', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--seed', type=int, default=239)
parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str,
                    default=os.getcwd().split('\code2seq_torch')[0] + '\code2seq_torch\config\code2seq-py150k.yaml')

args = parser.parse_args()
data_dir = Path(args.data_dir)


def test_compressing_then_reading():
    limit = 10
    # limit = 0
    evals = __collect_asts(data_dir / 'python50k_eval.json', limit=limit)

    graphs_eval = ast_to_graph.__collect_all_ast_graphs(evals, args)

    vocab_size = 10

    vocab = TPE.learn_vocabulary(graphs_eval, vocab_size)
    data = [ast_to_graph.graph_to_ast(graph) for graph in graphs_eval]

    compressed_graphs_file = data_dir / 'python50k_eval_formatted_temp.json'
    ast_to_graph.write_asts_to_file(compressed_graphs_file, data)

    compressed_graphs = __collect_asts(compressed_graphs_file, limit=0)
    assert len(compressed_graphs) == len(graphs_eval)
    first_graph = ast_to_graph.create_graph(compressed_graphs[0], 0)
    assert nx.algorithms.isomorphism.is_isomorphic(first_graph, graphs_eval[0])


def flatten(graphs, filename_to_write):
    graphs_eval = ast_to_graph.__collect_all_ast_graphs(graphs, args)
    data = [ast_to_graph.graph_to_ast(graph) for graph in graphs_eval]
    # return data
    ast_to_graph.write_asts_to_file(filename_to_write, data)


def test_flatting_then_reading_into_c2q_format():
    limit = 100
    para = True

    # create c2s for original
    evals = __collect_asts(data_dir / 'python50k_eval.json', limit=limit)
    # py_extractor.__collect_all_and_save(evals, args, './train.c2s', para=para)

    # flatten files
    compressed_graphs_file = data_dir / 'python50k_eval_flat_temp.json'
    flatten(__collect_asts(data_dir / 'python50k_eval.json', limit=limit), compressed_graphs_file)

    # create c2s for flattened
    flat_evals = __collect_asts(compressed_graphs_file, limit=0)
    # py_extractor.__collect_all_and_save(flat_evals, args, './new/train.c2s', para=para)

    c2s = py_extractor.collect_all(evals, args, para=para)
    flat_c2s = py_extractor.collect_all(flat_evals, args, para=para)

    assert set(c2s) == set(flat_c2s)


def test_read_flat_into_c2s():
    limit = 100
    para = True

    # flatten original
    compressed_graphs_file = data_dir / 'python50k_eval_flat_temp.json'
    flatten(__collect_asts(data_dir / 'python50k_eval.json', limit=limit), compressed_graphs_file)

    #create c2s for flattened
    flat_evals = __collect_asts(compressed_graphs_file, limit=0)
    py_extractor.__collect_all_and_save(flat_evals, args, './new/train.c2s', para=para)

    config = cast(DictConfig, OmegaConf.load(args.config))
    data_module = PathContextDataModule('./new', config.data)
    assert 'FunctionDef' in data_module.vocabulary.node_to_id
    assert 'If' in data_module.vocabulary.node_to_id
    assert 'Call' in data_module.vocabulary.node_to_id


test_read_flat_into_c2s()

test_flatting_then_reading_into_c2q_format()

test_compressing_then_reading()
