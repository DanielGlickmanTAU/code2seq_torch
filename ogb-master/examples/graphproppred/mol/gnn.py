import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from conv import GNN_node, GNN_node_Virtualnode


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 num_transformer_layers=0, feed_forward_dim=1024):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        # self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        self.num_transformer_layers = num_transformer_layers
        if num_transformer_layers:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=4, dim_feedforward=feed_forward_dim,norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.num_transformer_layers:
            h_node = self.forward_transformer(batched_data, h_node)
        # shape (num_graphs, out_dim)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

    def forward_transformer(self, batched_data, h_node):
        # change from batched_data(shape num_nodes in batch, emb_dim), to list where each item is of shape (#num_nodes in *graph*, emb_dim)
        # todo check torch_geometric.utils.to_dense_batch
        h_node_batch, distances_batched = self.split_into_graphs(batched_data, h_node)
        transformer_result = []
        for x in h_node_batch:
            #unsqueeze(1) -> transformer needs batch size in second dim by default
            bla = self.transformer(x.unsqueeze(1))
            ## [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            # graph_attn_bias = graph_attn_bias + spatial_pos_bias

            transformer_result.append(bla.squeeze(1))
        # back to original dim
        h_node = torch.cat(transformer_result, dim=0)
        return h_node

    def split_into_graphs(self, batched_data, h_node):
        graph_end_indexes = torch.unique_consecutive(batched_data.batch, return_counts=True)[1]
        graph_end_indexes_as_list = [x.item() for x in graph_end_indexes]

        h_node_batched = torch.split(h_node, graph_end_indexes_as_list)
        distances_batched = [torch.tensor(x) for x in batched_data.distances]

        return h_node_batched, distances_batched


if __name__ == '__main__':
    GNN(num_tasks=10)
