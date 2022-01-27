from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones


class MyTransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, encoder_layers: list, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        #         self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = encoder_layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, masks: list[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i, mod in enumerate(self.layers):
            mask = masks[i] if masks and len(masks) > i else None
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
