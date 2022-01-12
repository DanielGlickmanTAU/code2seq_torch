import gc
import argparse
import multiprocessing
from pathlib import Path
import os
from typing import cast

from omegaconf import DictConfig, OmegaConf
from sklearn import model_selection

from data import compression_args
from data.ast_conversion import ast_to_graph, TPE
import data.py150k_extractor as py_extractor

parser = compression_args.get_compressor_argparser()

args = parser.parse_args()
data_dir = Path(args.data_dir)

para = True
should_zip = False
out_files = []

limit = args.limit
vocab_size = args.vocab_size
max_word_joins = args.max_word_joins
merging_2_value_nodes = args.merging_value_nodes


def args_to_path_suffix(limit, vocab_size, max_word_joins, merging_2_value_nodes):
    return f'{vocab_size}_{max_word_joins}{("_" + str(limit)) if limit else ""}_Vmerge{merging_2_value_nodes}'


path_suffix = args_to_path_suffix(limit, vocab_size, max_word_joins, merging_2_value_nodes)
compressed_c2s_dir = Path(
    f'../../out_python/compressed_{path_suffix}')
#####
# compressed_c2s_dir = Path(f'../../out_python/compressed')

Path(compressed_c2s_dir).mkdir(exist_ok=True)
config = cast(DictConfig, OmegaConf.load(args.config))

eval = ast_to_graph.collect_all_functions(data_dir / 'python50k_eval.json', args, limit=limit // 2, n_cores=1)
functions = ast_to_graph.collect_all_functions(data_dir / 'python100k_train.json', args, limit=limit, n_cores=1)

#######
vocab = TPE.learn_vocabulary(eval + functions, vocab_size, max_word_joins, merging_2_value_nodes=merging_2_value_nodes)

joins_path = compressed_c2s_dir / f'vocab_{path_suffix}'
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
    py_extractor.new_collect_all_and_save(split, output_file, args,para=para)
    del split
    gc.collect()
    out_files.append(str(output_file))
print(out_files)
if should_zip:
    zip_name = f'py_c2s_compressed_{vocab_size}_{max_word_joins}.zip'
    what_in_zip = compressed_c2s_dir / '*'
    what_to_delete = compressed_c2s_dir / '*.c2s'
    os.system(f'zip  {zip_name} {what_in_zip} && rm {what_to_delete}')

print('done')