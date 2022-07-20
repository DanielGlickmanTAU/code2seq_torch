import torch

import global_config


class MaskedOperation(torch.nn.Module):
    def __init__(self, torch_operation):
        super().__init__()
        self.operation = torch_operation

    def forward(self, src: torch.Tensor, padding_mask: torch.Tensor):
        if not global_config.masked_norm:
            return self.operation(src)
        B, N, D = src.shape
        assert padding_mask.shape == (B, N)
        # avoid issues with copying, in place operations etc...
        x1 = src + 0
        x1[padding_mask] = self.operation(x1[padding_mask])
        return x1
