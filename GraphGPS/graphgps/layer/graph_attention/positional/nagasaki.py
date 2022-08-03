import torch
import torch_geometric

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.graph_attention.positional import positional_utils
from graphgps.layer.graph_attention.positional.PositionMultiHeadAttention import PositionMultiHeadAttention
# import torch_geometric.nn as nn
import torch.nn as nn


# diffusion edge reducer
class Nagasaki(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout, nagasaki_config):
        super().__init__()

        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        self.att = PositionMultiHeadAttention(dim_h, num_heads, edge_dim=edge_dim, dropout=dropout,
                                              batch_first=True, edge_reduction=nagasaki_config.edge_reduction)

    def forward(self, batch, h, mask):
        stacks = batch.edges
        atten_out, atten_weights = self.att(h, stacks, attn_mask=~mask)
        return atten_out, atten_weights
