from graphgps.layer.graph_attention.positional import positional_utils
from graphgps.layer.graph_attention.positional.MultiHeadAttention import MultiHeadAttention, PositionAttention

import torch


class Nagasaki(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, nagasaki_config, input_stacks=1):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        if nagasaki_config.content_attention_only:
            self._pos_attention = None
        else:
            position_attention_heads = num_heads * (input_stacks ** 2)
            self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=position_attention_heads,
                                                    edge_reduction=nagasaki_config.edge_reduction,
                                                    scale=nagasaki_config.scale_attention,
                                                    fuck_positional=nagasaki_config.fuck_positional)
        self.att = MultiHeadAttention(dim_h, num_heads, dropout=dropout,
                                      batch_first=True, merge_attention=nagasaki_config.merge_attention,
                                      content_only=nagasaki_config.content_attention_only)
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
    def __init__(self, dim_h, num_heads, dropout, nagasaki_config, input_stacks=1):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        if nagasaki_config.content_attention_only:
            self._pos_attention = None
        else:
            position_attention_heads = num_heads * input_stacks
            self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=position_attention_heads,
                                                    edge_reduction=nagasaki_config.edge_reduction,
                                                    scale=nagasaki_config.scale_attention,
                                                    fuck_positional=nagasaki_config.fuck_positional)
        self.att = MultiHeadAttention(dim_h, num_heads, dropout=dropout,
                                      batch_first=True, merge_attention=nagasaki_config.merge_attention,
                                      content_only=nagasaki_config.content_attention_only)
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
                # todo reshape properly..
                position_attention_weights = PositionAttention.reshape_positional_attention_to_joined_graph_attention(
                    position_attention_weights, self.input_stacks)

        # todo do not None
        position_attention_weights = None
        atten_out, atten_weights = self.att(h, batch.history[0], batch.history[0], position_attention_weights,
                                            attn_mask=~batch.history[1])

        return atten_out, atten_weights
