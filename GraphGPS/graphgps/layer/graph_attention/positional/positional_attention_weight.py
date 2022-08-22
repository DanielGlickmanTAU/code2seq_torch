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
                layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
            else:
                layers = [torch.nn.BatchNorm1d(input_dim), torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
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

    def __init__(self, steps: list, nhead=1, kernel=None, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.steps = steps
        self.kernel = kernel
        self.nhead = nhead
        assert len(self.steps) == len(set(self.steps)), f'duplicate power in {self.steps}'

    def forward(self, batch, mask, edge_weights=None):
        if edge_weights is not None:
            # edge_weights = edge_weights.squeeze(-1)
            if self.kernel:
                edge_weights = torch.sigmoid(edge_weights) if self.kernel == 'sigmoid' else torch.exp(
                    -(edge_weights ** 2)) if self.kernel == 'exp' else torch.exp(edge_weights)
            if edge_weights.dim() == 2:
                adj = to_dense_adj(batch.edge_index, batch.batch, edge_weights)
            else:
                adj = edge_weights
            assert adj.dim() == 4
            adj = adj.permute(0, 3, 1, 2)
        else:
            adj = to_dense_adj(batch.edge_index, batch.batch)

        if self.normalize:
            adj = self.to_P_matrix(adj)

        powers = self._calc_power(adj, self.steps)

        # if mask.dim() == 3:
        mask = pygraph_utils.dense_mask_to_attn_mask(mask)
        self_adj = torch.diag_embed((mask).float()).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        powers.insert(0, self_adj)
        stacks = torch.stack(powers, dim=-1)
        return stacks

    def to_P_matrix(self, A: torch.Tensor):
        # sum over rows
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


class MultiHeadAdjStackWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim_out, edge_model_type, ffn_layers, nhead, reduce):
        super().__init__()
        self.nhead = nhead
        self.reduce = reduce
        self.hidden_reducer_list = torch.nn.ModuleList([
            AdjStackAttentionWeights(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dim_out=1 if reduce else hidden_dim,
                ffn=edge_model_type,
                ffn_layers=ffn_layers)
            for _ in range(nhead)]
        )
        if reduce:
            self.hidden_reducer_combiner = AdjStackAttentionWeights(
                input_dim=self.nhead,
                hidden_dim=self.nhead * 2,
                dim_out=dim_out,
                ffn='mlp',
                ffn_layers=1
            )

    def forward(self, stacks, mask):
        reduced_edges = [self.hidden_reducer_list[i](stacks[:, i, :, :, :], mask) for i in range(stacks.shape[1])]
        if self.reduce:
            reduced_edges = torch.cat(reduced_edges, dim=-1)
            reduced_edges = self.hidden_reducer_combiner(reduced_edges, mask)
        else:
            reduced_edges = torch.stack(reduced_edges, dim=-1)
            reduced_edges = torch.sum(reduced_edges, dim=-1)
        return reduced_edges


class Diffuser(nn.Module):
    def __init__(self, dim_in, nagasaki_config):
        super().__init__()
        steps = nagasaki_config.steps
        if isinstance(steps, str):
            steps = list(eval(steps))
        num_stack = positional_utils.get_stacks_dim(nagasaki_config)
        edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        self.nhead = nagasaki_config.nhead
        if nagasaki_config.learn_edges_weight:
            self.edge_reducer = EdgeReducer(dim_in, hidden_dim=2 * dim_in, dim_out=self.nhead, dropout=0.,
                                            norm_output=nagasaki_config.bn_out)
        else:
            self.edge_reducer = None

        self.adj_stacker = AdjStack(steps, nhead=nagasaki_config.nhead, kernel=nagasaki_config.kernel,
                                    normalize=nagasaki_config.normalize)

        self.edge_mlp = MultiHeadAdjStackWeight(input_dim=num_stack,
                                                hidden_dim=edge_dim,
                                                dim_out=edge_dim,
                                                edge_model_type=nagasaki_config.edge_model_type,
                                                ffn_layers=nagasaki_config.ffn_layers,
                                                reduce=False, nhead=self.nhead)
        # AdjStackAttentionWeights(
        # input_dim=num_stack,
        # hidden_dim=edge_dim,
        # dim_out=edge_dim,
        # ffn=nagasaki_config.edge_model_type,
        # ffn_layers=nagasaki_config.ffn_layers)

        self.kernel = nagasaki_config.kernel
        self.two_diffusion = nagasaki_config.two_diffusion

        if nagasaki_config.two_diffusion:
            self.hidden_reducer = MultiHeadAdjStackWeight(
                input_dim=num_stack,
                hidden_dim=edge_dim,
                edge_model_type=nagasaki_config.edge_model_type,
                ffn_layers=nagasaki_config.ffn_layers,
                dim_out=self.nhead,
                nhead=self.nhead, reduce=True)

    def forward(self, batch):
        _, mask = get_dense_x_and_mask(batch.x, batch.batch)
        batch.mask = mask

        weighted_edges = None
        if self.edge_reducer:
            weighted_edges = self.edge_reducer(batch)

        stacks = self.adj_stacker(batch, mask, weighted_edges)

        if self.two_diffusion:
            reduced_edges = self.hidden_reducer(stacks, mask)

            stacks = self.adj_stacker(batch, mask, reduced_edges)

        edges = self.edge_mlp(stacks, mask)

        batch.edges = edges

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

        return e

    def __init__(self, in_dim, hidden_dim, dim_out, dropout, norm_output=False, **kwargs):
        super().__init__(**kwargs)

        self.A = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.C = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.edge_out_proj = torch_geometric.nn.Linear(hidden_dim, dim_out, bias=True)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.norm_output = norm_output
        if norm_output:
            self.bn_out = nn.BatchNorm1d(dim_out)
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
#useage:
# shape_edges = FakeReducer()(batch)
# shape_edges_full = to_dense_adj(batch.edge_index, batch.batch, shape_edges).squeeze(-1)
# # add self loops
# [g.fill_diagonal_(1) for g in shape_edges_full]
# visualization.draw_attention(batch[6].graph, 0, to_dense_adj(batch.edge_index, batch.batch, shape_edges).squeeze(-1)[6])
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
