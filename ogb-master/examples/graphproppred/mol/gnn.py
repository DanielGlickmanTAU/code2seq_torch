import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from conv import GNN_node, GNN_node_Virtualnode


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=6)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        # change from batched_data(shape num_nodes in batch, emb_dim), to list where each item is of shape (#num_nodes in *graph*, emb_dim)
        h_node_batch = self.split_into_graphs(batched_data, h_node)
        transformer_result = []
        for x in h_node_batch:
            bla = self.transformer(x.unsqueeze(0))
            transformer_result.append(bla.squeeze(0))

        #back to original dim
        transformer_h_node_batch = torch.cat(transformer_result, dim=0)
        #shape (num_graphs, out_dim)
        h_graph = self.pool(transformer_h_node_batch, batched_data.batch)

        return self.graph_pred_linear(h_graph)

    def split_into_graphs(self, batched_data, h_node):
        graph_end_indexes = torch.unique_consecutive(batched_data.batch, return_counts=True)[1]
        graph_end_indexes_as_list = [x.item() for x in graph_end_indexes]
        h_node_batched = torch.split(h_node, graph_end_indexes_as_list)
        return h_node_batched


if __name__ == '__main__':
    GNN(num_tasks=10)