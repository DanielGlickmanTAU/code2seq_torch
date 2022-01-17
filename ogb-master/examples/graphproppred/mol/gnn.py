import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from GNN_transformer import GNNTransformer


class GNN(torch.nn.Module):

    def __init__(self, args, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 num_transformer_layers=0, feed_forward_dim=1024):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.args = args
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_transformer = GNNTransformer(JK, args, drop_ratio, emb_dim, feed_forward_dim, gnn_type, num_layer,
                                              num_transformer_layers, residual, virtual_node)

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

    def forward(self, batched_data):
        h_node = self.gnn_transformer(batched_data)
        # shape (num_graphs, out_dim)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks=10)
