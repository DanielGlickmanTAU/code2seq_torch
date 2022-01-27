import torch
from torch import nn

from GraphDistanceBias import GraphDistanceBias
from conv import GNN_node_Virtualnode, GNN_node
from model.GraphTransformerEncoder import GraphTransformerEncoder
from pygraph_utils import split_into_graphs


class GNNTransformer(nn.Module):
    def __init__(self, JK, args, drop_ratio, emb_dim, feed_forward_dim, gnn_type, num_layer, num_transformer_layers,
                 residual, virtual_node, node_encoder=None):

        ### GNN to generate node embeddings

        self.emb_dim = emb_dim
        super(GNNTransformer, self).__init__()
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                                                 residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type, node_encoder=node_encoder)
        self.num_transformer_layers = num_transformer_layers
        if num_transformer_layers:
            self.transformer = GraphTransformerEncoder(args.attention_type, emb_dim, num_transformer_layers,
                                                       args.num_heads,
                                                       feed_forward_dim)
            self.distance_bias = GraphDistanceBias(args, num_heads=args.num_heads,
                                                   receptive_fields=args.receptive_fields)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.num_transformer_layers:
            h_node = self.forward_transformer(batched_data, h_node)
        return h_node

    def forward_transformer(self, batched_data, h_node):
        # change from batched_data(shape num_nodes in batch, emb_dim), to list where each item is of shape (#num_nodes in *graph*, emb_dim)
        # todo check torch_geometric.utils.to_dense_batch
        h_node_batch = split_into_graphs(batched_data, h_node)
        distances_batched = self.distance_bias(batched_data)
        transformer_result = []
        for x, distance_weights in zip(h_node_batch, distances_batched):
            # unsqueeze(1) -> transformer needs batch size in second dim by default
            bla = self.transformer(x.unsqueeze(1), masks=[distance_weights])

            transformer_result.append(bla.squeeze(1))
        # back to original dim, i.e pytorch geometric format
        h_node = torch.cat(transformer_result, dim=0)
        return h_node
