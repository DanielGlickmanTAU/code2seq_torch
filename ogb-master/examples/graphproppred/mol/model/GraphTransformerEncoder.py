from typing import Optional, List

from torch import Tensor
import torch.nn as nn

from code2seq.utils.compute import get_device
from model.MyTransformerEncoderLayer import MyTransformerEncoderLayer


class GraphTransformerEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--attention_type', type=str, default='content')

    __constants__ = ['norm']

    def __init__(self, attention_type, d_model, num_layers, num_head, num_adj_stacks, feed_forward_dim, dropout,
                 norm=None):
        super(GraphTransformerEncoder, self).__init__()
        #         self.layers = _get_clones(encoder_layer, num_layers)
        # TODO: dropout
        encoder_layers = nn.ModuleList([
            MyTransformerEncoderLayer(attention_type, d_model=d_model, nhead=num_head, num_adj_stacks=num_adj_stacks,
                                      dim_feedforward=feed_forward_dim, device=get_device(), dropout=dropout) for _ in
            range(num_layers)])
        self.layers = encoder_layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Tensor = None,
                src_key_padding_mask: Optional[Tensor] = None, adj_stack: Optional[Tensor] = None) -> Tensor:
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, adj_stack=adj_stack)

        if self.norm is not None:
            output = self.norm(output)

        return output
