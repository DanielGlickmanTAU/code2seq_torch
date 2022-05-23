import torch
from torch_geometric.nn import MessagePassing, GATv2Conv
import torch.nn.functional as F
from torch_geometric.utils import degree

from ogb.graphproppred.mol_encoder import BondEncoder


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, args, emb_dim, type='mol'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        hidden_dim = int(args.gin_conv_mlp_hidden_breath * emb_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, hidden_dim),
                                       torch.nn.BatchNorm1d(hidden_dim,
                                                            track_running_stats=args.conv_track_running_stats),
                                       torch.nn.ReLU(), torch.nn.Linear(hidden_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        if type == 'mol':
            self.edge_encoder = BondEncoder(emb_dim=emb_dim)
        elif type == 'code':
            self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = torch.zeros((edge_index.shape[-1], x.shape[-1]), device=x.device)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, type='mol'):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if type == 'mol':
            self.edge_encoder = BondEncoder(emb_dim=emb_dim)
        elif type == 'code':
            self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = torch.zeros((edge_index.shape[-1], x.shape[-1]), device=x.device)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, args, task, num_layer, emb_dim, node_encoder, drop_ratio=0.5, JK="last", residual=False,
                 gnn_type='gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.node_encoder = node_encoder

        self.task = task

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(args, emb_dim, self.task))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, self.task))
            elif gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(emb_dim, emb_dim, heads=2))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim,track_running_stats=args.conv_track_running_stats))

    def forward(self, batched_data):
        if self.task == 'code':
            x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch
            h_list = [self.node_encoder(x, node_depth.view(-1, ))]
        else:
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
            h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            # h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation = node_representation + h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
