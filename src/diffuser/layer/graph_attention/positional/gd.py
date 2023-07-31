from graphgps.layer.graph_attention.positional import positional_utils
from graphgps.layer.graph_attention.positional.MultiHeadAttention import MultiHeadAttention, PositionAttention

import torch


class GD(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, gd_config, input_stacks=1):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(gd_config)
        if gd_config.content_attention_only:
            self._pos_attention = None
        else:
            position_attention_heads = num_heads * (input_stacks ** 2)
            self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=position_attention_heads,
                                                    edge_reduction=gd_config.edge_reduction,
                                                    scale=gd_config.scale_attention,
                                                    ignore_positional=gd_config.ignore_positional)
        self.att = MultiHeadAttention(dim_h, num_heads, dropout=dropout,
                                      batch_first=True, merge_attention=gd_config.merge_attention,
                                      content_only=gd_config.content_attention_only)
        self.input_stacks = input_stacks

    def forward(self, batch, h, mask):
        # Diffuser forward sets this and saves in batch.
        dense_mask = batch.mask
        assert dense_mask.dim() == 3

        stacks = batch.edges

        position_attention_weights = None
        if self._pos_attention:
            position_attention_weights = self._pos_attention(stacks, dense_mask)
            if self.input_stacks > 1:
                position_attention_weights = PositionAttention.reshape_positional_attention_to_joined_graph_attention(
                    position_attention_weights, self.input_stacks)

        atten_out, atten_weights = self.att(h, h, h, position_attention_weights, attn_mask=~mask)

        return atten_out, atten_weights


class PatternAttention(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, gd_config, input_stacks=1):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(gd_config)
        if gd_config.content_attention_only:
            self._pos_attention = None
        else:
            position_attention_heads = num_heads * input_stacks
            self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=position_attention_heads,
                                                    edge_reduction=gd_config.edge_reduction,
                                                    scale=gd_config.scale_attention,
                                                    ignore_positional=gd_config.ignore_positional)
        self.att = MultiHeadAttention(dim_h, num_heads, dropout=dropout,
                                      batch_first=True, merge_attention=gd_config.merge_attention,
                                      content_only=gd_config.content_attention_only)
        self.input_stacks = input_stacks

    def forward(self, batch, h, mask):
        # Diffuser forward sets this and saves in batch.
        dense_mask = batch.mask
        assert dense_mask.dim() == 3

        stacks = batch.edges

        position_attention_weights = None
        if self._pos_attention:
            position_attention_weights = self._pos_attention(stacks, dense_mask)
            if self.input_stacks > 1:
                position_attention_weights = PositionAttention.reshape_positional_attention_to_cross_graph_attention(
                    position_attention_weights, self.input_stacks)

        history = batch.history[0]
        history_mask = batch.history[1]
        atten_out, atten_weights = self.att(h, history, history, position_attention_weights,
                                            attn_mask=~history_mask)

        return atten_out, atten_weights
