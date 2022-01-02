import argparse

import os

import multiprocessing


def get_compressor_argparser():
    parser = argparse.ArgumentParser()
    data_dir = os.getcwd().split('data')[0] + '/data/python'
    parser.add_argument('--data_dir', default=data_dir, type=str)
    parser.add_argument('--valid_p', type=float, default=0.2)
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)
    parser.add_argument('--use_method_name', type=bool, default=True)
    parser.add_argument('--use_nums', type=bool, default=True)
    parser.add_argument('--output_dir', default='out_python', type=str)
    parser.add_argument('--n_jobs', type=int, default=min(multiprocessing.cpu_count(), 8))
    parser.add_argument('--seed', type=int, default=239)
    parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str,
                        default=os.getcwd().split('code2seq_torch')[0] + '/code2seq_torch/config/code2seq-py150k.yaml')
    parser.add_argument('--max_word_joins', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max_context_per_method_c2s', type=int, default=400)
    parser.add_argument('--merging_value_nodes', type=lambda x: x and x.lower() == 'true', default=True)

    return parser
