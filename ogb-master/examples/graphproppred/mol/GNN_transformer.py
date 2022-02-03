import torch
import torch_geometric
from torch import nn

from GraphDistanceBias import GraphDistanceBias
from conv import GNN_node_Virtualnode, GNN_node
from model.GraphTransformerEncoder import GraphTransformerEncoder
import pygraph_utils
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
                                                       args.num_heads, len(args.adj_stacks),
                                                       feed_forward_dim)
            # self.distance_bias = GraphDistanceBias(args, num_heads=args.num_heads,
            #                                        receptive_fields=args.receptive_fields)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.num_transformer_layers:
            h_node = self.forward_transformer(batched_data, h_node)
        return h_node

    def forward_transformer(self, batched_data, h_node):
        # (n_graph,max_nodes_in_graph,emb_dim), (n_graph,max_nodes_in_graph)
        h_node_batch, mask = pygraph_utils.get_dense_x_and_mask(h_node, batched_data.batch)
        adj_stack = pygraph_utils.get_dense_adjstack(batched_data.adj_stack, batched_data.batch)
        # h_node_batch = split_into_graphs(batched_data, h_node)

        transformer_result = []

        # for x, distance_weights, adj_stack in zip(h_node_batch, distances_batched, batched_data.adj_stack):
        # adj_stack = torch.tensor(adj_stack, device=x.device)
        # unsqueeze(1) -> transformer needs batch size in second dim by default
        x = self.transformer(h_node_batch, mask=mask, adj_stack=adj_stack)

        # back to original dim, i.e pytorch geometric format
        spare_x = pygraph_utils.get_spare_x(x, mask)
        assert spare_x.shape == h_node.shape
        # h_node = torch.cat(transformer_result, dim=0)
        return spare_x
