import argparse
import time

from code2seq.utils import compute
from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStack

compute.get_torch()
import datetime
import os
import torch
import logging

from yacs.config import CfgNode

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from custom.config import parse_args
# from graphgps.config.wandb2_config import set_cfg_wandb
from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    try:
        if cfg.train.auto_resume:
            os.makedirs(cfg.run_dir, exist_ok=True)
        else:
            makedirs_rm_exist(cfg.run_dir)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # Load cmd line args
    args = argparse.ArgumentParser().parse_args()
    args.cfg_file = 'configs/graph/row-coloring-laplace.yaml'
    args.words_per_row = 3
    args.num_rows = 5
    args.atom_set = 8
    args.max_examples = 100
    args.num_unique_colors = 2
    args.num_unique_atoms = 1
    args.make_prob_of_row_half = True
    args.opts = []
    # args.wandb.use = False
    # Load config file
    set_cfg(cfg)

    load_cfg(cfg, args)
    for key, value in args.__dict__.items():
        try:
            # this handle nested props like optim.base_lr
            keys = key.split('.')
            cfg_dict = cfg
            for k in keys[:-1]:
                cfg_dict = cfg_dict[k]
            cfg_dict[key] = value
            print(f'overwrite {key}:{value}')
        except Exception:
            print(f'fail to overwrite {key}:{value}')

    cfg.device = str(compute.get_device())

    # cfg.dataset.split_index = split_index
    # cfg.seed = seed
    # cfg.run_id = run_id
    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline
    loaders = create_loader()
    loggers = create_logger()
    model = create_model()

    args.adj_stacks = range(20)
    args.use_distance_bias = False
    args.normalize = True
    data = loaders[0].dataset[0]
    stacks = AdjStack(args)(data)['adj_stack']
    stacks = stacks.transpose(1, 2, 0)
    stacks = torch.tensor(stacks)
    stacks_batch = torch.stack([stacks])
