import math
from typing import Optional, Tuple

import torch
from torch.nn import Parameter
from torch.nn.functional import *
from torch import Tensor
from torch.nn.functional import _in_projection_packed, _in_projection
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.overrides import has_torch_function, handle_torch_function

import pygraph_utils
from model.positional.PositionMultiHeadAttention import multi_head_positional_attention
from model.positional.positional_attention_weight import AdjStackAttentionWeights


class ContentMultiheadAttention(torch.nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, args, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, use_distance_bias=False,
                 num_adj_stacks=0) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ContentMultiheadAttention, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.gating = args.gating
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
        if use_distance_bias:
            self.positional_bias = AdjStackAttentionWeights(num_adj_stacks, num_heads)

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
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, adj_stack: Optional[Tensor] = None) -> \
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
            # v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

            B, Nt, E = q.shape
            if self.gating:
                print('diving by sqrt when gating.. not sure we want this')
            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q, k.transpose(-2, -1))
            assert not attn.isinf().any(), attn
            assert not attn.isnan().any(), attn

            if self.use_distance_bias:
                attn = attn + self._positional_bias_f(adj_stack)

            attn_mask = pygraph_utils.reshape_attention_mask_to_multihead(attn_mask, self.num_heads)
            new_attn = torch.zeros_like(attn, device=attn.device)
            new_attn[~attn_mask] = attn[~attn_mask]

            attn_output, attn_output_weights = multi_head_positional_attention(
                v, new_attn, self.embed_dim, self.num_heads,
                self.bias_k, self.bias_v,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, scale_by_sqrt_n=False, gating=self.gating)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def _positional_bias_f(self, adj_stack):
        distance_bias = self.positional_bias(adj_stack)
        # collapse togeter first(batch) and second(head) dim
        distance_bias = distance_bias.reshape(distance_bias.shape[0] * distance_bias.shape[1],
                                              distance_bias.shape[-1], distance_bias.shape[-1])
        return distance_bias
