from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_nagasaki(cfg):
    cfg.nagasaki = CN()

    cfg.nagasaki.steps = None
    cfg.nagasaki.edge_model_type = 'mlp'  # /linear/bn-l
    cfg.nagasaki.ffn_hidden_multiplier = 2
    cfg.nagasaki.ffn_layers = 1

    cfg.nagasaki.edge_reduction = 'bn-mlp'


register_config('cfg_nagasaki', set_cfg_nagasaki)
