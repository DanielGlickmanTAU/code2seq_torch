from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_nagasaki(cfg):
    cfg.nagasaki = CN()

    cfg.nagasaki.steps = None
    cfg.nagasaki.edge_model_type = 'mlp'  # /linear/bn-l
    cfg.nagasaki.ffn_hidden_multiplier = 1
    cfg.nagasaki.ffn_layers = 1
    cfg.nagasaki.edge_reducer_hidden_dim = 2

    cfg.nagasaki.edge_reduction = 'bn-mlp'
    cfg.nagasaki.learn_edges_weight = False
    cfg.nagasaki.skip_stacking_ratio = 0.
    cfg.nagasaki.symmetric_edge_reduce = False
    cfg.nagasaki.two_diffusion = False
    cfg.nagasaki.normalize = True
    cfg.nagasaki.kernel = 'sigmoid'
    cfg.nagasaki.bn_out = False
    cfg.nagasaki.nhead = 1
    cfg.nagasaki.merge_attention = None
    cfg.nagasaki.add_cls = False
    cfg.nagasaki.project_diagonal = False
    cfg.nagasaki.scale_attention = False
    cfg.nagasaki.skip_cls_pooling = False

    cfg.nagasaki.type = 'transformer'


register_config('cfg_nagasaki', set_cfg_nagasaki)
