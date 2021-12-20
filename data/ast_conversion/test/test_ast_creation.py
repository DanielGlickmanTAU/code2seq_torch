import unittest
import argparse
import multiprocessing
from pathlib import Path
import os
from typing import cast

from omegaconf import DictConfig, OmegaConf

from code2seq.data.path_context_data_module import PathContextDataModule
from data.ast_conversion import ast_to_graph, TPE
import data.py150k_extractor as py_extractor

parser = argparse.ArgumentParser()
data_dir = os.getcwd().split('data')[0] + '/data/python'
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
                    default=os.getcwd().split('code2seq_torch')[0] + '/code2seq_torch/config/code2seq-py150k.yaml')
parser.add_argument('--max_node_joins', type=int, default=999)

args = parser.parse_args()
data_dir = Path(args.data_dir)

para = True
max_word_joins = 999


def flatten(graphs, filename_to_write):
    graphs_eval = ast_to_graph.collect_all_ast_graphs(graphs, args)
    ast_to_graph.write_asts_to_file(filename_to_write, graphs_eval)


class TestCompression(unittest.TestCase):

    def test_learning_vocab(self):
        limit = 10
        evals = ast_to_graph.collect_asts(data_dir / 'python50k_eval.json', limit=limit)

        graphs_eval = ast_to_graph.collect_all_ast_graphs(evals, args)

        vocab_size = 10

        vocab = TPE.learn_vocabulary(graphs_eval, vocab_size, max_word_joins)
        assert ('AttributeLoad', 'attr') in vocab

    def test_limit_vocab_length(self):
        uncompressed_c2s_dir = Path('./uncomp1')
        Path(uncompressed_c2s_dir).mkdir(exist_ok=True)
        limit = 10
        vocab_size = 10
        max_word_joins = 1

        functions = ast_to_graph.collect_all_functions(data_dir / 'python50k_eval.json', args, limit=limit)
        vocab = TPE.learn_vocabulary(functions, vocab_size, max_word_joins)
        paths_compressed = py_extractor.collect_all(functions, args, True)

        for path in paths_compressed:
            for context in path.split(' '):
                for node in context.split(py_extractor.token_separator):
                    assert node.count(
                        TPE.vocab_separator) <= max_word_joins, \
                        f'word {node} was merged from {node.count(TPE.vocab_separator) + 1} nodes'

    def test_flatting_then_reading_into_c2q_format(self):
        limit = 100
        compressed_graphs_file = data_dir / 'python50k_eval_flat_temp.json'

        # flatten files
        flatten(
            ast_to_graph.collect_asts(data_dir / 'python50k_eval.json', limit=limit),
            compressed_graphs_file
        )

        # create c2s for flattened
        flat_evals = ast_to_graph.collect_asts(compressed_graphs_file, limit=0)
        flat_c2s = py_extractor.collect_all(flat_evals, args, para=para)

        # create c2s for original
        evals = ast_to_graph.collect_asts(data_dir / 'python50k_eval.json', limit=limit)
        c2s = py_extractor.collect_all(evals, args, para=para)

        assert set(c2s) == set(flat_c2s)

    def test_read_flat_into_c2s(self):
        limit = 100
        output_dir = './new'
        Path(output_dir).mkdir(exist_ok=True)
        compressed_graphs_file = data_dir / 'python50k_eval_flat_temp.json'

        # flatten original
        asts = ast_to_graph.collect_asts(data_dir / 'python50k_eval.json', limit=limit)
        flatten(asts, compressed_graphs_file)

        # create c2s for flattened
        flat_evals = ast_to_graph.collect_asts(compressed_graphs_file, limit=0)
        py_extractor.collect_all_and_save(flat_evals, args, '%s/train.c2s' % output_dir, para=para)

        config = cast(DictConfig, OmegaConf.load(args.config))
        data_module = PathContextDataModule('%s' % output_dir, config.data)
        assert 'FunctionDef' in data_module.vocabulary.node_to_id
        assert 'If' in data_module.vocabulary.node_to_id
        assert 'Call' in data_module.vocabulary.node_to_id

    def test_compressed_dataset(self):
        uncompressed_c2s_dir = Path('./uncomp1')
        compressed_c2s_dir = './comp2'
        Path(uncompressed_c2s_dir).mkdir(exist_ok=True)
        Path(compressed_c2s_dir).mkdir(exist_ok=True)
        config = cast(DictConfig, OmegaConf.load(args.config))
        limit = 100
        vocab_size = 100

        functions = ast_to_graph.collect_all_functions(data_dir / 'python50k_eval.json', args, limit=limit)
        paths = py_extractor.collect_all(functions, args, para)

        functions2 = ast_to_graph.collect_all_functions(data_dir / 'python50k_eval.json', args, limit=limit)
        vocab = TPE.learn_vocabulary(functions2, vocab_size, max_word_joins)
        paths_compressed = py_extractor.collect_all(functions2, args, True)

        assert len(paths_compressed) == len(paths)

        py_extractor.write_to_file('%s/train.c2s' % uncompressed_c2s_dir, paths)
        data_module = PathContextDataModule('./%s' % uncompressed_c2s_dir, config.data)

        py_extractor.write_to_file('./%s/train.c2s' % compressed_c2s_dir, paths_compressed)
        data_module_compressed = PathContextDataModule('./%s' % compressed_c2s_dir, config.data)

        assert data_module_compressed.vocabulary.label_to_id == data_module.vocabulary.label_to_id
        assert data_module_compressed.vocabulary.node_to_id != data_module.vocabulary.node_to_id
        assert None not in list(data_module_compressed.train_dataloader().dataset)


if __name__ == "__main__":
    unittest.main()
