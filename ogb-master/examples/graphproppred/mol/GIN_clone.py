from torch_geometric.nn import MessagePassing

from code2seq.utils import compute
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""


class GINNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        # in_dim = net_params['in_dim']
        in_dim = 3
        # hidden_dim = net_params['hidden_dim']
        hidden_dim = 110
        # n_classes = net_params['n_classes']
        n_classes = 2
        # dropout = net_params['dropout']
        dropout = 0.
        # self.n_layers = net_params['L']
        self.n_layers = 4
        # n_mlp_layers = net_params['n_mlp_GIN']  # GIN
        n_mlp_layers = 2
        learn_eps = True
        neighbor_aggr_type = 'sum'
        # batch_norm = net_params['batch_norm']
        batch_norm = True
        # residual = net_params['residual']
        residual = True
        self.n_classes = n_classes
        self.device = compute.get_device()

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        self.embedding_h = nn.Embedding(in_dim, hidden_dim)

        for layer in range(self.n_layers):
            # mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(GINLayer(args,hidden_dim))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        h = self.embedding_h(x)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](x=h,edge_index=edge_index,edge_attr=None)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            score_over_layer += self.linears_prediction[i](h)

        return score_over_layer

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""


class GINLayer(MessagePassing):
    def __init__(self, args, emb_dim, type='mol'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINLayer, self).__init__(aggr="add")

        hidden_dim = int(args.gin_conv_mlp_hidden_breath * emb_dim)
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim),
        #                                torch.nn.ReLU(), torch.nn.Linear(hidden_dim, emb_dim))
        self.mlp = MLP(2, emb_dim, emb_dim, emb_dim)

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bn_node_h = nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = torch.zeros((edge_index.shape[-1], x.shape[-1]), device=x.device)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        h = self.bn_node_h(out)  # batch normalization

        h = F.relu(h)  # non-linear activation

        h = x + h  # residual connection

        # h = F.dropout(h, self.dropout, training=self.training)

        return h

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# class GINLayer(nn.Module):
#     def __init__(self, apply_func, aggr_type, dropout, batch_norm, residual=False, init_eps=0, learn_eps=False):
#         super().__init__()
#         self.apply_func = apply_func
#
#         if aggr_type == 'sum':
#             self._reducer = fn.sum
#         elif aggr_type == 'max':
#             self._reducer = fn.max
#         elif aggr_type == 'mean':
#             self._reducer = fn.mean
#         else:
#             raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
#
#         self.batch_norm = batch_norm
#         self.residual = residual
#         self.dropout = dropout
#
#         in_dim = apply_func.mlp.input_dim
#         out_dim = apply_func.mlp.output_dim
#
#         if in_dim != out_dim:
#             self.residual = False
#
#         # to specify whether eps is trainable or not.
#         if learn_eps:
#             self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
#         else:
#             self.register_buffer('eps', torch.FloatTensor([init_eps]))
#
#         self.bn_node_h = nn.BatchNorm1d(out_dim)
#
#     def forward(self, g, h):
#         h_in = h  # for residual connection
#
#         g = g.local_var()
#         g.ndata['h'] = h
#         g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
#         h = (1 + self.eps) * h + g.ndata['neigh']
#         if self.apply_func is not None:
#             h = self.apply_func(h)
#
#         if self.batch_norm:
#             h = self.bn_node_h(h)  # batch normalization
#
#         h = F.relu(h)  # non-linear activation
#
#         if self.residual:
#             h = h_in + h  # residual connection
#
#         h = F.dropout(h, self.dropout, training=self.training)
#
#         return h


class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                i_ = self.linears[i](h)
                h = F.relu(self.batch_norms[i](i_))
            return self.linears[-1](h)
