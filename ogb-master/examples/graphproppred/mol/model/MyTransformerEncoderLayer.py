from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, Dropout, LayerNorm, BatchNorm1d
from torch.nn.modules.transformer import _get_activation_fn

from model.ContentMultiHeadAttention import ContentMultiheadAttention
from model.masked_operation import MaskedOperation
from model.positional.PositionMultiHeadAttention import PositionMultiHeadAttention


class MyTransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, args, attention_type, d_model, nhead, num_adj_stacks=None, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_distance_bias=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyTransformerEncoderLayer, self).__init__()
        if attention_type == 'content':
            self.attention_layer = ContentMultiheadAttention(args, d_model, nhead, use_distance_bias=use_distance_bias,
                                                             num_adj_stacks=num_adj_stacks,
                                                             dropout=dropout, batch_first=batch_first,
                                                             **factory_kwargs)
        elif attention_type == 'position':
            assert num_adj_stacks
            self.attention_layer = PositionMultiHeadAttention(args, d_model, nhead, num_adj_stacks, dropout=dropout,
                                                              batch_first=batch_first,
                                                              **factory_kwargs)
        else:
            raise Exception(f'{attention_type} attention type unsupported')
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        norm1 = LayerNorm(d_model, eps=layer_norm_eps,
                          **factory_kwargs) if args.attention_norm_type == 'layer' else BatchNorm1d(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        norm2 = LayerNorm(d_model, eps=layer_norm_eps,
                          **factory_kwargs) if args.ff_norm_type == 'layer' else BatchNorm1d(
            d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm1 = MaskedOperation(norm1)
        self.norm2 = MaskedOperation(norm2)

        self.use_batch_norm_in_transformer_mlp = args.use_batch_norm_in_transformer_mlp
        if self.use_batch_norm_in_transformer_mlp:
            self.batch_norm = MaskedOperation(BatchNorm1d(dim_feedforward, eps=layer_norm_eps, **factory_kwargs))

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MyTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, adj_stack: Optional[Tensor] = None) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x, src_key_padding_mask), src_mask, None, adj_stack)
            x = x + self._ff_block(self.norm2(x, src_key_padding_mask), src_key_padding_mask)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, None, adj_stack), src_key_padding_mask)
            x = self.norm2(x + self._ff_block(x, src_key_padding_mask), src_key_padding_mask)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  adj_stack: Optional[Tensor] = None) -> Tensor:
        if isinstance(self.attention_layer, PositionMultiHeadAttention):
            x = self.attention_layer(x, adj_stack,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False)[0]
        else:
            x = self.attention_layer(x, x, x,
                                     adj_stack=adj_stack,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor, src_key_padding_mask) -> Tensor:
        x1 = self.linear1(x)
        if self.use_batch_norm_in_transformer_mlp:
            x1 = self.batch_norm(x1, src_key_padding_mask)
        x = self.linear2(self.dropout(self.activation(x1)))
        return self.dropout2(x)
