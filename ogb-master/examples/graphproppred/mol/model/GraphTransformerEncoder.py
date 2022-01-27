from typing import Optional, List

from torch import Tensor
from torch.nn import Module

from model.MyTransformerEncoderLayer import MyTransformerEncoderLayer


class GraphTransformerEncoder(Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--attention_type', type=str, default='content')

    __constants__ = ['norm']

    def __init__(self, attention_type, d_model, num_layers, num_head, feed_forward_dim, norm=None):
        super(GraphTransformerEncoder, self).__init__()
        #         self.layers = _get_clones(encoder_layer, num_layers)
        encoder_layers = [MyTransformerEncoderLayer(attention_type, d_model=d_model, nhead=num_head,
                                                    dim_feedforward=feed_forward_dim) for _ in
                          range(num_layers)]
        self.layers = encoder_layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, masks: List[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for i, mod in enumerate(self.layers):
            mask = masks[i] if masks and len(masks) > i else None
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
