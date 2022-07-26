from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_nagasaki(cfg):
    cfg.nagasaki = CN()

    cfg.nagasaki.steps = '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'


register_config('cfg_nagasaki', set_cfg_nagasaki)
