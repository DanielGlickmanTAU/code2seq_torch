import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from graphgps.layer.graph_attention.positional.PositionMultiHeadAttention import PositionMultiHeadAttention
from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights


class Nagasaki(torch.nn.Module):
    def __init__(self, dim_h, num_heads, dropout):
        super().__init__()
        steps = [1, 2, 3, 4]
        self.adj_stacker = AdjStack(steps)
        # edge_reducer = AdjStackAttentionWeights(num_adj_stacks=len(steps), num_heads=n_heads, ffn=True)
        self.att = PositionMultiHeadAttention(dim_h, num_heads, num_adj_stacks=len(steps) + 1, dropout=dropout,
                                              batch_first=True)

    def forward(self, batch, h, mask):
        # h_dense, mask = to_dense_batch(h, batch.batch)
        stacks = self.adj_stacker(batch, mask)
        atten_out, atten_weights = self.att(h, stacks, attn_mask=~mask)
        return atten_out, atten_weights
