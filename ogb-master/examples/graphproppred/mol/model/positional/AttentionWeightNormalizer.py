import torch


class AttentionWeightNormalizer(torch.nn.Module):
    def __init__(self, gating):
        super().__init__()
        if gating:
            self.func = torch.nn.Sigmoid()
        else:
            self.func = torch.nn.Softmax(dim=-1)

    def forward(self, attention_weights, attn_mask):
        def fix_nans(attn):
            not_nan = ~attn.isnan()
            attn_new = torch.zeros_like(attn, device=attn.device)
            attn_new[not_nan] = attn[not_nan]
            return attn_new

        if attn_mask is not None:
            assert attn_mask.min() == float('-inf') or not attn_mask.any()
            assert attn_mask.max() == 0.
            attention_weights = attention_weights + attn_mask

        return fix_nans(self.func(attention_weights))
