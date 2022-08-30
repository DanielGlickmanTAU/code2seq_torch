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
                 add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, use_distance_bias=False,
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

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.use_distance_bias = use_distance_bias

        self.normalizer = AttentionWeightNormalizer(gating=False)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
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

        if not self._qkv_same_embed_dim:
            raise Exception('should not happen')
        else:
            tgt_len, bsz, embed_dim = value.shape
            head_dim = embed_dim // self.num_heads

            q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
            q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

            B, Nt, E = q.shape

            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q, k.transpose(-2, -1))
            assert not attn.isinf().any(), attn
            assert not attn.isnan().any(), attn



            attn_mask = pygraph_utils.reshape_attention_mask_to_multihead(attn_mask, self.num_heads)
            attn_output, attn_output_weights = multi_head_positional_attention(
                v, attn, self.embed_dim, self.num_heads,
                self.bias_k, self.bias_v,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, scale_by_sqrt_n=False, normalizer=self.normalizer)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
