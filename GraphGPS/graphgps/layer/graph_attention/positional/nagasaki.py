from graphgps.layer.graph_attention.positional import positional_utils
from graphgps.layer.graph_attention.positional.MultiHeadAttention import MultiHeadAttention, PositionAttention

import torch


class Nagasaki(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, nagasaki_config):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        if nagasaki_config.content_attention_only:
            self._pos_attention = None
        else:
            self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=num_heads,
                                                    edge_reduction=nagasaki_config.edge_reduction,
                                                    scale=nagasaki_config.scale_attention)
        self.att = MultiHeadAttention(dim_h, num_heads, edge_dim=edge_dim, dropout=dropout,
                                      batch_first=True, edge_reduction=nagasaki_config.edge_reduction,
                                      merge_attention=nagasaki_config.merge_attention or nagasaki_config.content_attention_only,
                                      scale=nagasaki_config.scale_attention)

    def forward(self, batch, h, mask):
        # Diffuser forward sets this and saves in batch.
        dense_mask = batch.mask
        assert dense_mask.dim() == 3

        stacks = batch.edges

        position_attention_weights = self._pos_attention(stacks, dense_mask) if self._pos_attention else None

        atten_out, atten_weights = self.att(h, position_attention_weights, attn_mask=~mask)

        return atten_out, atten_weights
