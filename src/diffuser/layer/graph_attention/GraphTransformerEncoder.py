from typing import Optional, List

from torch import Tensor
import torch.nn as nn

from arg_parse_utils import bool_
from model.MyTransformerEncoderLayer import MyTransformerEncoderLayer
from model.positional.positional_attention_weight import AdjStackAttentionWeights

from GraphGPS.graphgps.utils import get_device


class GraphTransformerEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--attention_type', type=str, default='content')
        parser.add_argument('--gating', type=bool_, default=False,
                            help='if true, use element wise sigmoid instead of softmax in attention')
        # But default it is on, which is opposed to the original transformer but is used by many
        parser.add_argument('--norm_first', type=bool_, default=True,
                            help='apply layer norm before or after self attention and feedforward. Default True')

        parser.add_argument('--attention_norm_type', type=str, default='layer',
                            help='use layer or batch norm. see also norm_first')
        parser.add_argument('--ff_norm_type', type=str, default='layer',
                            help='use layer or batch norm. see also norm_first')
        parser.add_argument('--use_batch_norm_in_transformer_mlp', type=bool_, default=False)

        AdjStackAttentionWeights.add_args(parser)

    __constants__ = ['norm']

    def __init__(self, args, attention_type, d_model, num_layers, num_head, num_adj_stacks, feed_forward_dim, dropout,
                 norm=None, use_distance_bias=False):
        super(GraphTransformerEncoder, self).__init__()
        encoder_layers = nn.ModuleList([
            MyTransformerEncoderLayer(args, attention_type, d_model=d_model, nhead=num_head,
                                      num_adj_stacks=num_adj_stacks,
                                      norm_first=args.norm_first,
                                      dim_feedforward=feed_forward_dim, device=get_device(), dropout=dropout,
                                      use_distance_bias=use_distance_bias) for _ in
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
