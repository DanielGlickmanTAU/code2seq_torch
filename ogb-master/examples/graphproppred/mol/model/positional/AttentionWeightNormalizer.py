import torch


class AttentionWeightNormalizer(torch.nn.Module):
    def __init__(self, gating):
        super().__init__()
        if gating:
            self.func = torch.nn.Sigmoid()
        else:
            self.func = torch.nn.Softmax(dim=-1)

    def forward(self, attention_weights):
        return self.func(attention_weights)
