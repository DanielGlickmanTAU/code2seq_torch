from typing import Optional, Tuple

import torch
from torch import Tensor

from graphgps.layer.graph_attention.positional.AttentionWeightNormalizer import AttentionWeightNormalizer
from examples.graphproppred.mol import pygraph_utils
from torch.nn.functional import linear, dropout


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

) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    tgt_len, bsz, embed_dim = value.shape
    src_len = tgt_len
    head_dim = embed_dim // num_heads
    _assertions(bias_k, bias_v, embed_dim, embed_dim_to_check, head_dim, num_heads,
                use_separate_proj_weight, key_padding_mask)

    "expect attention weights to be of shape (batch *num_head,tgt_len,tgt_len)"
    assert attention_weights.shape == (bsz * num_heads, src_len, src_len)
    attn_mask = prep_attention_mask(attn_mask, bsz, num_heads, src_len, tgt_len)


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


# def def_scale(attention_weights):
#     # scale = old_mask.sum(dim=-1, keepdim=True).sqrt()
#     # attention_weights =
#     # q = q / math.sqrt(E)
#     attention_weights = attention_weights
#     return attention_weights


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
            correct_3d_size = {(bsz * num_heads, tgt_len, src_len), (bsz * num_heads, 1, src_len)}

            if attn_mask.shape not in correct_3d_size:
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
