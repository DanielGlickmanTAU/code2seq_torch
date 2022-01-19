import re
from code2seq.utils import compute

torch = compute.get_torch()
from argparse import ArgumentParser
from typing import cast

import torch
from commode_utils.common import print_config
from omegaconf import DictConfig, OmegaConf

from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.model import Code2Seq
from code2seq.utils.common import filter_warnings
from code2seq.utils.test import test
from code2seq.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("mode", help="Mode to run script", choices=["train", "test"], default='train')
    # arg_parser.add_argument("--c", "--config", help="Path to YAML configuration file", type=str)
    arg_parser.add_argument("--config", help="Path to YAML configuration file", type=str)
    arg_parser.add_argument("--train_on_val", help="Train on validation set for debugging", type=bool, default=False)
    arg_parser.add_argument("--max_num_examples",
                            help="Limit the number of training/validaition examples. Mostly used for debugging",
                            type=int, default=0)
    arg_parser.add_argument('--embedding_size',
                            help='embedding size of model. If None(default) will take from config file, if not, will overwrite it.',
                            type=int, default=None)
    arg_parser.add_argument('--data_folder',
                            help='location of {train,val,test}.c2s. If None(default), will take from config file. If not, will overwrite it.',
                            type=str, default=None)

    return arg_parser


def train_code2seq(config: DictConfig, args):
    filter_warnings()

    if config.print_config:
        print_config(config, fields=["model", "data", "train", "optimizer"])

    # Load data module
    data_module = PathContextDataModule(config.data_folder, config.data, limit=args.max_num_examples)

    if args.train_on_val:
        data_module._train = 'val'

    # Load model
    code2seq = Code2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    train(code2seq, data_module, config)


def test_code2seq(config: DictConfig):
    filter_warnings()

    # Load data module
    data_module = PathContextDataModule(config.data_folder, config.data)

    # Load model
    code2seq = Code2Seq.load_from_checkpoint(config.checkpoint, map_location=torch.device("cpu"))

    test(code2seq, data_module, config.seed)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    __config = cast(DictConfig, OmegaConf.load(__args.config))
    if __args.data_folder:
        __config.data_folder = __args.data_folder
    if __args.embedding_size:
        __config.model.embedding_size = __args.embedding_size
    print(__config.model.embedding_size)

    if __args.mode == "train":
        train_code2seq(__config, __args)
    else:
        test_code2seq(__config)
