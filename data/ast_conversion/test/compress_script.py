import gc
import argparse
import multiprocessing
from pathlib import Path
import os
from typing import cast

from omegaconf import DictConfig, OmegaConf
from sklearn import model_selection

from data.ast_conversion import ast_to_graph, TPE
import data.py150k_extractor as py_extractor

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

args = parser.parse_args()
data_dir = Path(args.data_dir)

para = True
should_zip = True
out_files = []

limit = args.limit
vocab_size = args.vocab_size
max_word_joins = args.max_word_joins

compressed_c2s_dir = Path(
    f'../../out_python/compressed_{vocab_size}_{max_word_joins}{("_" + str(limit)) if limit else ""}')
#####
# compressed_c2s_dir = Path(f'../../out_python/compressed')

Path(compressed_c2s_dir).mkdir(exist_ok=True)
config = cast(DictConfig, OmegaConf.load(args.config))

eval = ast_to_graph.collect_all_functions(data_dir / 'python50k_eval.json', args, limit=limit // 2)
functions = ast_to_graph.collect_all_functions(data_dir / 'python100k_train.json', args, limit=limit)

#######
vocab = TPE.learn_vocabulary(eval + functions, vocab_size, max_word_joins)

joins_path = compressed_c2s_dir / f'vocab_{vocab_size}_{max_word_joins}'
print(joins_path)
open(joins_path, 'w+').write(str(vocab))

train, valid = model_selection.train_test_split(
    functions,
    test_size=args.valid_p,
)
test = eval

for split_name, split in zip(
        ('train', 'val', 'test'),
        # ('test',),
        (train, valid, test),
        # (test,),

):
    output_file = compressed_c2s_dir / f'{split_name}.c2s'
    # py_extractor.collect_all_and_save(split, args, output_file, para=True)
    py_extractor.new_collect_all_and_save(split, args, output_file)
    del split
    gc.collect()
    out_files.append(str(output_file))
print(out_files)
if should_zip:
    zip_name = f'py_c2s_compressed_{vocab_size}_{max_word_joins}.zip'
    what_in_zip = compressed_c2s_dir / '*'
    what_to_delete = compressed_c2s_dir / '*.c2s'
    os.system(f'zip  {zip_name} {what_in_zip} && rm {what_to_delete}')
