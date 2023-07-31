from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_gd(cfg):
    cfg.gd = CN()

    cfg.gd.steps = None
    cfg.gd.edge_model_type = 'bn-mlp'  # /linear/bn-l
    cfg.gd.ffn_hidden_multiplier = 1
    cfg.gd.ffn_layers = 1
    cfg.gd.edge_reducer_hidden_dim = 2

    cfg.gd.edge_reduction = 'linear'
    cfg.gd.learn_edges_weight = False
    cfg.gd.skip_stacking_ratio = 0.
    cfg.gd.symmetric_edge_reduce = False
    cfg.gd.normalize = True
    cfg.gd.kernel = 'sigmoid'
    cfg.gd.bn_out = False
    cfg.gd.nhead = 1
    cfg.gd.merge_attention = None
    cfg.gd.content_attention_only = False
    cfg.gd.add_cls = False
    cfg.gd.project_diagonal = False
    cfg.gd.scale_attention = False
    cfg.gd.skip_cls_pooling = False

    cfg.gd.type = 'transformer'
    cfg.gd.ignore_positional = False
    cfg.gd.interleave_self_cross_attn = False


register_config('cfg_gd', set_cfg_gd)
