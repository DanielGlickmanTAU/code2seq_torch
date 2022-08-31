import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import _in_projection_packed
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from graphgps.layer.graph_attention.positional.AttentionWeightNormalizer import AttentionWeightNormalizer

from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStackAttentionWeights

from graphgps.layer.graph_attention.positional.attention import multi_head_positional_attention
from examples.graphproppred.mol import pygraph_utils


class ContentMultiheadAttention(torch.nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ContentMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.attention = ContentAttention(embed_dim, num_heads, bias=True, kdim=None, vdim=None, device=None,
                                          dtype=None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.normalizer = AttentionWeightNormalizer(False)

        self._reset_parameters()

        if not self._qkv_same_embed_dim:
            raise Exception('should not happen')

    def _reset_parameters(self):
        constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> \
            Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn, v = self.attention(query, key, value)

        attn_mask = pygraph_utils.reshape_attention_mask_to_multihead(attn_mask, self.num_heads)
        attn_output, attn_output_weights = multi_head_positional_attention(
            v, attn, self.embed_dim, self.num_heads,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, normalizer=self.normalizer)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class ContentAttention(torch.nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, bias=True, kdim=None, vdim=None, device=None, dtype=None,
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ContentAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        assert self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value):
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        attn = self._compute_attn(k, q)
        return attn, v

    def _compute_attn(self, k, q):
        tgt_len, bsz, embed_dim = k.shape
        head_dim = embed_dim // self.num_heads
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        assert not attn.isinf().any(), attn
        assert not attn.isnan().any(), attn
        return attn
