import torch
import torch_geometric
from torch import nn
# import global_config
# from arg_parse_utils import bool_
from torch_geometric.utils import to_dense_adj
from examples.graphproppred.mol.pygraph_utils import get_dense_x_and_mask

from examples.graphproppred.mol import pygraph_utils
from graphgps.layer.graph_attention.positional import positional_utils


class AdjStackAttentionWeights(torch.nn.Module):

    def __init__(self, input_dim, dim_out, hidden_dim, ffn, ffn_layers=1):
        super(AdjStackAttentionWeights, self).__init__()
        self.num_adj_stacks = input_dim
        self.num_heads = dim_out
        if ffn == 'bn-linear':
            self.weight = torch.nn.Sequential(
                torch.nn.BatchNorm1d(input_dim),
                torch.nn.Linear(input_dim, dim_out),

            )
        elif ffn == 'bn-mlp' or ffn == 'mlp':
            if ffn == 'mlp':
                layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim)]
            else:
                layers = [torch.nn.BatchNorm1d(input_dim), torch.nn.Linear(input_dim, hidden_dim)]
            layers.append(torch.nn.ReLU())
            for _ in range(ffn_layers - 1):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_dim, dim_out))

            self.weight = torch.nn.Sequential(*layers)

        elif ffn == 'linear':
            self.weight = nn.Linear(in_features=input_dim, out_features=dim_out)
        else:
            raise ValueError(f'nagasaki does not support edge forward model of type {ffn}')

    # stacks shape is (batch,n,n,num_adj_stacks)
    # mask shape is (batch,n).
    # returns (batch,n,n,num_heads)
    def forward(self, stacks: torch.Tensor, mask):
        b, n, n1, num_stacks = stacks.shape
        assert num_stacks == self.num_adj_stacks
        assert mask.dim() == 3

        adj_weights = torch.zeros((b, n1, n, self.num_heads), device=stacks.device)
        adj_weights[mask] = self.weight(stacks[mask].view(-1, num_stacks))
        assert adj_weights.shape == (b, n, n1, self.num_heads)
        return adj_weights


class AdjStack(torch.nn.Module):

    def __init__(self, steps: list, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.steps = steps
        assert len(self.steps) == len(set(self.steps)), f'duplicate power in {self.steps}'

    def forward(self, batch, mask, edge_weights=None):
        if edge_weights is not None:
            edge_weights = edge_weights.squeeze(-1)
            if edge_weights.dim() == 3:
                adj = edge_weights
            else:
                assert edge_weights.dim() == 1, f'supporting only single weighted edges, got weights of shape {edge_weights.shape}'
                adj = to_dense_adj(batch.edge_index, batch.batch, edge_weights)
        else:
            adj = to_dense_adj(batch.edge_index, batch.batch)
        adj = self.to_P_matrix(adj)

        powers = self._calc_power(adj, self.steps) if self.normalize else adj

        # if mask.dim() == 3:
        mask = pygraph_utils.dense_mask_to_attn_mask(mask)
        self_adj = torch.diag_embed((mask).float())
        powers.insert(0, self_adj)
        stacks = torch.stack(powers, dim=-1)
        return stacks

    def to_P_matrix(self, A: torch.Tensor):
        D = A.sum(dim=-1, keepdim=True)
        # if all entries are zero, we want to avoid dividing by zero.
        # set it to any number, as the entries in A are 0 anyway so A/D will be 0
        D[D == 0] = 1
        return A / D

    def _calc_power(self, adj, steps):
        powers = []
        if steps == list(range(min(steps), max(steps) + 1)):
            # Efficient way if ksteps are a consecutive sequence (most of the time the case)
            Pk = adj.clone().detach().matrix_power(min(steps))
            powers.append(Pk)
            for k in range(min(steps), max(steps)):
                Pk = Pk @ adj
                powers.append(Pk)
        else:
            for k in steps:
                Pk = torch.diagonal(adj.matrix_power(k), dim1=-2, dim2=-1)
                powers.append(Pk)
        return powers


class Diffuser(nn.Module):
    def __init__(self, dim_in, nagasaki_config):
        super().__init__()
        steps = nagasaki_config.steps
        if isinstance(steps, str):
            steps = list(eval(steps))
        num_stack = positional_utils.get_stacks_dim(nagasaki_config)
        edge_dim = positional_utils.get_edge_dim(nagasaki_config)

        if nagasaki_config.learn_edges_weight:
            self.edge_reducer = EdgeReducer(dim_in, hidden_dim=2 * dim_in, dropout=0.,
                                            norm_output=nagasaki_config.bn_out)
        else:
            self.edge_reducer = None

        self.adj_stacker = AdjStack(steps, normalize=nagasaki_config.normalize)

        self.edge_mlp = AdjStackAttentionWeights(
            input_dim=num_stack,
            hidden_dim=edge_dim,
            dim_out=edge_dim,
            ffn=nagasaki_config.edge_model_type,
            ffn_layers=nagasaki_config.ffn_layers)

        self.kernel = nagasaki_config.kernel
        self.two_diffusion = nagasaki_config.two_diffusion
        if nagasaki_config.two_diffusion:
            self.hidden_edge_mlp = AdjStackAttentionWeights(
                input_dim=num_stack,
                hidden_dim=edge_dim,
                dim_out=1,
                ffn=nagasaki_config.edge_model_type,
                ffn_layers=nagasaki_config.ffn_layers)

    def forward(self, batch):
        # h_dense, mask2 = to_dense_batch(batch.x, batch.batch)
        # adj = to_dense_adj(batch.edge_index, batch.batch)

        if 'mask' in batch.keys:
            mask = batch.mask
        else:
            _, mask = get_dense_x_and_mask(batch.x, batch.batch)
            batch.mask = mask

        weighted_edges = None
        if self.edge_reducer:
            weighted_edges = self.edge_reducer(batch)
            weighted_edges = torch.sigmoid(weighted_edges) if self.kernel == 'sigmoid' else torch.exp(
                -(weighted_edges ** 2))

            # shape_edges = FakeReducer()(batch)
            # shape_edges_full = to_dense_adj(batch.edge_index, batch.batch, shape_edges).squeeze(-1)
            # # add self loops
            # [g.fill_diagonal_(1) for g in shape_edges_full]

        stacks = self.adj_stacker(batch, mask, weighted_edges)

        if self.two_diffusion:
            reduced_edges = self.hidden_edge_mlp(stacks, mask)
            stacks = self.adj_stacker(batch, mask, reduced_edges)

        edges = self.edge_mlp(stacks, mask)

        batch.edges = edges
        # visualization.draw_attention(batch[6].graph, 0, to_dense_adj(batch.edge_index, batch.batch, shape_edges).squeeze(-1)[6])

        return batch


class EdgeReducer(torch_geometric.nn.conv.MessagePassing):

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        # symetric distance, map i and j the same
        Ax = self.A(x)
        Ce = self.C(e)

        e = self.propagate(edge_index,
                           Ce=Ce,
                           e=e, Ax=Ax,
                           )
        # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # e = torch.nn.functional.dropout(e, self.dropout, training=self.training)

        return e

    def __init__(self, in_dim, hidden_dim, dropout, norm_output=False, **kwargs):
        super().__init__(**kwargs)

        self.A = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.C = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.edge_out_proj = torch_geometric.nn.Linear(hidden_dim, 1, bias=True)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.norm_output = norm_output
        if norm_output:
            self.bn_out = nn.BatchNorm1d(1)
        self.dropout = dropout

    def message(self, Ax_i, Ax_j, Ce):
        e_ij = Ax_i + Ax_j + Ce
        e_ij = torch.nn.functional.relu(self.bn(e_ij))
        e_ij = self.edge_out_proj(e_ij)
        if self.norm_output:
            e_ij = torch.nn.functional.relu(self.bn_out(e_ij))

        sigma_ij = e_ij
        # sigma_ij = torch.sigmoid(e_ij)
        return sigma_ij

    def aggregate(self, e):
        return e


# returns if 1./0 if (*real*) edge is inside shape
class FakeReducer(torch_geometric.nn.conv.MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x_id = batch.g_id
        return self.propagate(batch.edge_index, X=x_id.unsqueeze(1))

    def message(self, X_i, X_j):
        return (X_i == X_j).float()

    def aggregate(self, e):
        return e
