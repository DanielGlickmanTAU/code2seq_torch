from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from graphgps.layer.graph_attention.positional.attention import multi_head_positional_attention
from examples.graphproppred.mol import pygraph_utils
from graphgps.layer.graph_attention.positional.AttentionWeightNormalizer import AttentionWeightNormalizer
from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStackAttentionWeights


class PositionMultiHeadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, num_adj_stacks, dropout=0., bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PositionMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        assert self._qkv_same_embed_dim

        self.gating: bool = False
        self.scale_positional_attention = False
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        # self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(1 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.positional_bias = AdjStackAttentionWeights(num_adj_stacks, num_heads)
        self.normalizer = AttentionWeightNormalizer(gating=self.gating)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, value: Tensor, adj_stack: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            value = value.transpose(1, 0)

        attention_weights = self.positional_bias(stacks=adj_stack, mask=attn_mask)
        b, heads, n1, n2 = attention_weights.shape
        # attention_weights = attention_weights.contiguous().view(b * self.num_heads, n1, n1)
        attention_weights = attention_weights.reshape(b * self.num_heads, n1, n1)
        # (n,batch,d)
        value = linear(value, self.in_proj_weight, self.in_proj_bias)
        attn_mask = pygraph_utils.reshape_attention_mask_to_multihead(attn_mask, self.num_heads)

        attn_output, attn_output_weights = multi_head_positional_attention(
            value, attention_weights, self.embed_dim, self.num_heads,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,

            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, scale_by_sqrt_n=self.scale_positional_attention, normalizer=self.normalizer)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
