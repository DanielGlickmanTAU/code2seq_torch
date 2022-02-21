from torch import nn

from conv import GNN_node
from model.GraphTransformerEncoder import GraphTransformerEncoder
import pygraph_utils


class GNNTransformer(nn.Module):
    def __init__(self, JK, args, drop_ratio, emb_dim, feed_forward_dim, gnn_type, num_layer, num_transformer_layers,
                 residual, virtual_node, node_encoder, task):

        ### GNN to generate node embeddings

        self.emb_dim = emb_dim
        super(GNNTransformer, self).__init__()
        if virtual_node:
            raise Exception('not supported')
        else:
            self.gnn_node = GNN_node(task, num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type, node_encoder=node_encoder)
        self.num_transformer_layers = num_transformer_layers
        if num_transformer_layers:
            self.transformer = GraphTransformerEncoder(args.attention_type, emb_dim, num_transformer_layers,
                                                       args.num_heads, len(args.adj_stacks),
                                                       feed_forward_dim, args.transformer_encoder_dropout)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.num_transformer_layers:
            h_node = self.forward_transformer(batched_data, h_node)
        return h_node

    def forward_transformer(self, batched_data, h_node):
        # (n_graph,max_nodes_in_graph,emb_dim), (n_graph,max_nodes_in_graph)
        h_node_batch, mask = pygraph_utils.get_dense_x_and_mask(h_node, batched_data.batch)
        adj_stack = pygraph_utils.get_dense_adjstack(batched_data.adj_stack, batched_data.batch)

        x = self.transformer(h_node_batch, mask=mask, adj_stack=adj_stack)

        # back to original dim, i.e pytorch geometric format
        spare_x = pygraph_utils.get_spare_x(x, mask)
        assert spare_x.shape == h_node.shape
        return spare_x
