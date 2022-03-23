from torch import nn

import global_config
from conv import GNN_node
from model.GraphTransformerEncoder import GraphTransformerEncoder
import pygraph_utils


class GNNTransformer(nn.Module):
    def __init__(self, JK, args, drop_ratio, emb_dim, feed_forward_dim, gnn_type, num_layer, num_transformer_layers,
                 residual, virtual_node, node_encoder, task):

        ### GNN to generate node embeddings

        self.emb_dim = emb_dim
        super(GNNTransformer, self).__init__()
        self.mask_far_away_nodes = args.mask_far_away_nodes
        if virtual_node:
            raise Exception('not supported')
        else:
            self.gnn_node = GNN_node(args, task, num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type, node_encoder=node_encoder)
        self.num_transformer_layers = num_transformer_layers
        if num_transformer_layers:
            self.transformer = GraphTransformerEncoder(args, args.attention_type, emb_dim, num_transformer_layers,
                                                       args.num_heads, len(args.adj_stacks),
                                                       feed_forward_dim, args.transformer_encoder_dropout,
                                                       use_distance_bias=args.use_distance_bias)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.num_transformer_layers:
            h_node = self.forward_transformer(batched_data, h_node)
        return h_node

    def forward_transformer(self, batched_data, h_node):
        # (n_graph,max_nodes_in_graph,emb_dim), (n_graph,max_nodes_in_graph)
        h_node_batch, original_mask = pygraph_utils.get_dense_x_and_mask(h_node, batched_data.batch)
        adj_stack = pygraph_utils.get_dense_adjstack(batched_data.adj_stack, batched_data.batch)
        if self.mask_far_away_nodes:
            # size (B,N,N)
            mask = (adj_stack.sum(dim=1) == 0)
        else:
            mask = original_mask

        # size (B,N)
        padding_mask = ~original_mask[:, 0]

        x = self.transformer(h_node_batch, mask=mask, adj_stack=adj_stack, src_key_padding_mask=padding_mask)

        # back to original dim, i.e pytorch geometric format
        spare_x = pygraph_utils.get_spare_x(x, original_mask)
        assert spare_x.shape == h_node.shape
        return spare_x
