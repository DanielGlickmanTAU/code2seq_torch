import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import linear, softmax, dropout
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

import pygraph_utils
from model.positional.AttentionWeightNormalizer import AttentionWeightNormalizer
from model.positional.positional_attention_weight import AdjStackAttentionWeights


class PositionMultiHeadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, args, embed_dim, num_heads, num_adj_stacks, dropout=0., bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        self.args = args
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PositionMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        assert self._qkv_same_embed_dim

        self.gating: bool = args.gating
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
        assert self._qkv_same_embed_dim

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
            attn_mask=attn_mask, scale_by_sqrt_n=self.args.scale_positional_attention, normalizer=self.normalizer)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_positional_attention(
        value: Tensor,
        attention_weights: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        normalizer: AttentionWeightNormalizer,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        scale_by_sqrt_n: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    tgt_len, bsz, embed_dim = value.shape
    src_len = tgt_len
    head_dim = embed_dim // num_heads
    _assertions(bias_k, bias_v, embed_dim, embed_dim_to_check, head_dim, num_heads,
                use_separate_proj_weight, key_padding_mask)

    "expect attention weights to be of shape (batch *num_head,tgt_len,tgt_len)"
    assert attention_weights.shape == (bsz * num_heads, src_len, src_len)
    assert attn_mask.shape == attention_weights.shape
    old_mask = attn_mask
    attn_mask = prep_attention_mask(attn_mask, bsz, num_heads, src_len, tgt_len)
    if scale_by_sqrt_n:
        attention_weights = _scale(attention_weights)

    # (batch*num_head, n , d/head)
    v = pygraph_utils.reshape_to_multihead(value, num_heads)

    assert v.shape[0] == num_heads * bsz
    assert tgt_len == v.shape[1]
    assert embed_dim == out_proj_weight.shape[-1]

    attn, attn_output = weighted_average(v, attention_weights, attn_mask, training, dropout_p, normalizer)
    attn_output = project_heads(attn_output, bsz, embed_dim, out_proj_bias, out_proj_weight, tgt_len)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def _scale(attention_weights):
    # scale = old_mask.sum(dim=-1, keepdim=True).sqrt()
    # attention_weights =
    # q = q / math.sqrt(E)
    attention_weights = attention_weights
    return attention_weights


def project_heads(attn_output, bsz, embed_dim, out_proj_bias, out_proj_weight, tgt_len):
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    return attn_output


def weighted_average(values, attention_weights, attn_mask, training, dropout_p, normalizer):
    attn = normalizer(attention_weights, attn_mask)

    # adjust dropout
    if not training:
        dropout_p = 0.0

    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    attn_output = torch.bmm(attn, values)
    return attn, attn_output


def prep_attention_mask(attn_mask, bsz, num_heads, src_len, tgt_len):
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            print("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            # convert mask to float
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            # new_attn_mask.masked_fill_(attn_mask, -1e6)
            attn_mask = new_attn_mask
    return attn_mask


def _assertions(bias_k, bias_v, embed_dim, embed_dim_to_check, head_dim, num_heads,
                use_separate_proj_weight, key_padding_mask):
    assert key_padding_mask is None

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        raise Exception
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    assert bias_k is None
    assert bias_v is None
    assert not use_separate_proj_weight
