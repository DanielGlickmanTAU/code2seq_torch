from graphgps.layer.graph_attention.positional import positional_utils
from graphgps.layer.graph_attention.positional.MultiHeadAttention import MultiHeadAttention

import torch


class Nagasaki(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, nagasaki_config):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        self.att = MultiHeadAttention(dim_h, num_heads, edge_dim=edge_dim, dropout=dropout,
                                      batch_first=True, edge_reduction=nagasaki_config.edge_reduction,
                                      merge_attention=nagasaki_config.merge_attention,
                                      scale=nagasaki_config.scale_attention)

    def forward(self, batch, h, mask):
        assert mask.dim() == 3
        stacks = batch.edges
        atten_out, atten_weights = self.att(h, stacks, attn_mask=mask)

        return atten_out, atten_weights
