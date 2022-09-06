from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from graphgps.layer.graph_attention.ContentMultiHeadAttention import ContentAttention
from graphgps.layer.graph_attention.positional.attention import multi_head_positional_attention
from examples.graphproppred.mol import pygraph_utils
from graphgps.layer.graph_attention.positional.AttentionWeightNormalizer import AttentionWeightNormalizer
from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStackAttentionWeights


class MultiHeadAttention(torch.nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, edge_dim, dropout=0., bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 edge_reduction='bn-mlp', merge_attention=None
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        assert self._qkv_same_embed_dim

        assert merge_attention is None or merge_attention == 'plus' or merge_attention == 'gate'
        self.merge_attention = merge_attention
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self.merge_attention:
            self.in_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))

        if bias and not self.merge_attention:
            self.in_proj_bias = Parameter(torch.empty(1 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self._pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=num_heads, edge_reduction=edge_reduction)
        self.content_attention = ContentAttention(embed_dim, num_heads, bias=True, kdim=None, vdim=None, device=None,
                                                  dtype=None) if merge_attention else None
        self.normalizer = AttentionWeightNormalizer(False)

        self._reset_parameters()

    def _reset_parameters(self):
        if not self.merge_attention:
            xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            if not self.merge_attention:
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

        position_attention_weights = self._pos_attention(adj_stack, attn_mask)

        if self.merge_attention:
            content_attention_weights, value = self.content_attention(value, value, value)
            content_attention_weights = torch.clamp(content_attention_weights, min=-5, max=5)
            position_attention_weights = torch.clamp(position_attention_weights, min=-5, max=5)
        else:
            content_attention_weights, value = None, linear(value, self.in_proj_weight, self.in_proj_bias)

        if self.merge_attention == 'plus':
            attention_weights = position_attention_weights + content_attention_weights
        elif self.merge_attention == 'gate':
            attention_weights = torch.log(torch.sigmoid(position_attention_weights)) + content_attention_weights
            print('sanity check')
            exp_sigmoid = torch.exp(content_attention_weights) * torch.sigmoid(position_attention_weights)
            assert (torch.softmax(attention_weights, dim=-1) - (exp_sigmoid) / (exp_sigmoid).sum(dim=-1,
                                                                                                 keepdim=True)).max() < 1e-4
        else:
            attention_weights = position_attention_weights
        # (n,batch,d)
        attn_mask = pygraph_utils.dense_mask_to_attn_mask(attn_mask)
        attn_mask = ~attn_mask
        attn_mask = pygraph_utils.reshape_attention_mask_to_multihead(attn_mask, self.num_heads)
        attn_output, attn_output_weights = multi_head_positional_attention(
            value, attention_weights, self.embed_dim, self.num_heads,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,

            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, normalizer=self.normalizer)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class PositionAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, num_heads, edge_dim, edge_reduction='bn-mlp') -> None:
        self.num_heads = num_heads
        super(PositionAttention, self).__init__()
        self.positional_bias = AdjStackAttentionWeights(edge_dim, dim_out=num_heads, ffn=edge_reduction,
                                                        hidden_dim=edge_dim * 1,
                                                        ffn_layers=0)

    def forward(self, adj_stack, attn_mask):
        attention_weights = self.positional_bias(stacks=adj_stack, mask=attn_mask)
        b, n1, n2, heads = attention_weights.shape
        assert heads == self.num_heads
        attention_weights = attention_weights.reshape(b * self.num_heads, n1, n1)
        return attention_weights
