from random import random

import torch
import torch_geometric
from torch import nn
from torch_geometric.utils import to_dense_adj
from examples.graphproppred.mol.pygraph_utils import get_dense_x_and_mask

from examples.graphproppred.mol import pygraph_utils
from graphgps.layer.graph_attention.positional import positional_utils


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class SoftmaxLinear(nn.Module):
    def __init__(self, input_dim, dim_out):
        super().__init__()
        self.weight = nn.Linear(in_features=input_dim, out_features=dim_out).weight

    def forward(self, stacks):
        weight_softmax = self.weight.softmax(-1)
        return torch.nn.functional.linear(stacks, weight_softmax)


class AdjStackAttentionWeights(torch.nn.Module):
    def __init__(self, input_dim, dim_out, hidden_dim, ffn, ffn_layers=1):
        super(AdjStackAttentionWeights, self).__init__()
        self.num_adj_stacks = input_dim
        self.num_heads = dim_out
        if not ffn or ffn == 'None':
            self.weight = Identity()
        if ffn == 'bn-linear':
            self.weight = torch.nn.Sequential(
                torch.nn.BatchNorm1d(input_dim),
                torch.nn.Linear(input_dim, dim_out),

            )
        elif ffn == 'softmax-linear':
            self.weight = SoftmaxLinear(input_dim, dim_out)
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
        elif ffn == 'res-mlp':
            layers = [torch.nn.BatchNorm1d(input_dim)]
            for _ in range(ffn_layers):
                linear_bn = [torch.nn.Linear(input_dim, input_dim), torch.nn.BatchNorm1d(input_dim)]
                layers.append(Residual(torch.nn.Sequential(*linear_bn)))
                layers.append(torch.nn.ReLU())

            layers.append(torch.nn.Linear(input_dim, dim_out))
            self.weight = torch.nn.Sequential(*layers)
        elif ffn == 'res-net':
            layers = []
            for _ in range(ffn_layers):
                linear_bn = [torch.nn.BatchNorm1d(input_dim), torch.nn.ReLU(), torch.nn.Linear(input_dim, input_dim),
                             torch.nn.BatchNorm1d(input_dim), torch.nn.ReLU(), torch.nn.Linear(input_dim, input_dim)]
                layers.append(Residual(torch.nn.Sequential(*linear_bn)))

            layers.append(torch.nn.Linear(input_dim, dim_out))
            self.weight = torch.nn.Sequential(*layers)

        elif ffn == 'linear':
            self.weight = nn.Linear(in_features=input_dim, out_features=dim_out)
        else:
            raise ValueError(f'nagasaki does not support edge forward model of type {ffn}')

    # stacks shape is (batch,n,n,num_adj_stacks)
    # mask shape is (batch,n,n).
    # returns (batch,n,n,num_heads)
    def forward(self, stacks: torch.Tensor, mask):
        b, n, n1, num_stacks = stacks.shape
        assert num_stacks == self.num_adj_stacks
        assert mask.dim() == 3

        adj_weights = torch.zeros((b, n, n1, self.num_heads), device=stacks.device)
        adj_weights[mask] = self.weight(stacks[mask].view(-1, num_stacks))
        return adj_weights


def to_P_matrix(A: torch.Tensor):
    # sum over rows
    D = A.sum(dim=-1, keepdim=True)
    # if all entries are zero, we want to avoid dividing by zero.
    # set it to any number, as the entries in A are 0 anyway so A/D will be 0
    D[D == 0] = 1
    return A / D


def _calc_power(adj, steps):
    steps = sorted(steps)
    powers = []
    # assert steps == list(range(min(steps), max(steps) + 1)), f'only consecutive sequences of power, got {steps}'
    assert len(steps) == len(set(steps)), 'found duplicate power'
    Pk = adj.matrix_power(min(steps))
    powers.append(Pk)
    for i in range(1, len(steps)):
        k = steps[i]
        prev_k = steps[i - 1]
        if prev_k + 1 == k:
            Pk = Pk @ adj
        elif prev_k * 2 == k:
            Pk = Pk @ Pk
        else:
            raise Exception('powers need to increase by one or doubled')
        powers.append(Pk)

    return powers


class AdjStack(torch.nn.Module):

    def __init__(self, steps: list, nhead=1, kernel=None, normalize: bool = True):
        super(AdjStack, self).__init__()
        self.normalize = normalize
        self.steps = steps
        self.kernel = kernel
        self.nhead = nhead
        assert len(self.steps) == len(set(self.steps)), f'duplicate power in {self.steps}'

    def forward(self, batch, mask, edge_weights=None):
        if edge_weights is not None:
            if self.kernel:
                edge_weights = torch.sigmoid(edge_weights) if self.kernel == 'sigmoid' \
                    else torch.exp(-(edge_weights ** 2)) if self.kernel == 'exp' \
                    else torch.exp(-(edge_weights ** 2) * 0.1) if self.kernel == 'exp-norm' \
                    else torch.exp(edge_weights - max(edge_weights)) if self.kernel == 'softmax' \
                    else None
            if edge_weights.dim() == 2:
                adj = to_dense_adj(batch.edge_index, batch.batch, edge_weights)
            else:
                adj = edge_weights
            assert adj.dim() == 4
            adj = adj.permute(0, 3, 1, 2)
        else:
            adj = to_dense_adj(batch.edge_index, batch.batch).unsqueeze(1)

        if self.normalize:
            adj = to_P_matrix(adj)

        powers = _calc_power(adj, self.steps)

        # if mask.dim() == 3:
        mask = pygraph_utils.dense_mask_to_attn_mask(mask)
        self_adj = torch.diag_embed((mask).float()).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        powers.insert(0, self_adj)
        stacks = torch.stack(powers, dim=-1)
        return stacks


class AdjStackGIN(torch.nn.Module):

    def __init__(self, steps: list, nhead=1):
        super().__init__()
        self.steps = steps
        self.nhead = nhead
        assert len(self.steps) == len(set(self.steps)), f'duplicate power in {self.steps}'
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, batch, mask, edge_weights=None):
        assert edge_weights is None
        adj = to_dense_adj(batch.edge_index, batch.batch).unsqueeze(1)

        adj[mask.unsqueeze(1)] = (1 + self.eps)
        adj = to_P_matrix(adj)

        powers = _calc_power(adj, self.steps)

        mask = pygraph_utils.dense_mask_to_attn_mask(mask)
        self_adj = torch.diag_embed((mask).float()).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        powers.insert(0, self_adj)
        stacks = torch.stack(powers, dim=-1)
        return stacks


class MultiHeadAdjStackWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim_out, edge_model_type, ffn_layers, nhead, reduce):
        super(MultiHeadAdjStackWeight, self).__init__()
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
            # hidden_reducer projected them, now we sum. This is like concating then projecting.
            reduced_edges = torch.sum(reduced_edges, dim=-1)
        return reduced_edges


class Diffuser(nn.Module):
    def __init__(self, dim_in, nagasaki_config):
        super(Diffuser, self).__init__()
        self.nhead = nagasaki_config.nhead
        self.kernel = nagasaki_config.kernel
        self.edge_dim = positional_utils.get_edge_dim(nagasaki_config)

        steps = nagasaki_config.steps
        if isinstance(steps, str):
            steps = list(eval(steps))
        num_stack = positional_utils.get_stacks_dim(nagasaki_config)

        if nagasaki_config.type == 'gin':
            assert not nagasaki_config.learn_edges_weight, 'should not learn weights with gin right now'
        if nagasaki_config.learn_edges_weight:
            self.edge_reducer = EdgeReducer(dim_in, hidden_dim=nagasaki_config.edge_reducer_hidden_dim * dim_in,
                                            dim_out=self.nhead, dropout=0.,
                                            symmetric=nagasaki_config.symmetric_edge_reduce)
        else:
            self.edge_reducer = None
        self.skip_stacking_ratio = nagasaki_config.skip_stacking_ratio

        self.adj_stacker = AdjStack(steps, nhead=nagasaki_config.nhead, kernel=nagasaki_config.kernel,
                                    normalize=nagasaki_config.normalize) if nagasaki_config.type != 'gin' else AdjStackGIN(
            steps, nagasaki_config.nhead)

        self.edge_mlp = MultiHeadAdjStackWeight(input_dim=num_stack,
                                                hidden_dim=self.edge_dim,
                                                dim_out=self.edge_dim,
                                                edge_model_type=nagasaki_config.edge_model_type,
                                                ffn_layers=nagasaki_config.ffn_layers,
                                                reduce=False, nhead=self.nhead)
        self.positional_embedding = nagasaki_config.project_diagonal
        if self.positional_embedding:
            num_stacks = len(steps) + 1
            self.positional_projection = nn.Sequential(
                *[torch.nn.BatchNorm1d(num_stacks), torch.nn.Linear(num_stacks, dim_in), torch.nn.ReLU()])

    def forward(self, batch):
        x, mask = get_dense_x_and_mask(batch.x, batch.batch)

        weighted_edges = self.edge_reducer(batch) if self.edge_reducer else None
        stacks = self.adj_stacker(batch, mask, weighted_edges)
        edges = self.edge_mlp(stacks, mask)

        if self.positional_embedding:
            self_edge = stacks.squeeze(1)[torch.diag_embed(pygraph_utils.dense_mask_to_attn_mask(mask))]
            pos_emb = self.positional_projection(self_edge)
            batch.x = batch.x + pos_emb

        batch.mask = mask
        batch.edges = edges
        return batch


class EdgeReducer(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, in_dim, hidden_dim, dim_out, dropout, symmetric):
        super(EdgeReducer, self).__init__()

        self.symmetric = symmetric
        self.A = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        if not self.symmetric:
            self.B = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.C = torch_geometric.nn.Linear(in_dim, hidden_dim, bias=True)
        self.edge_out_proj = torch_geometric.nn.Linear(hidden_dim, dim_out, bias=False)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if e is None:
            e = torch.zeros(edge_index.shape[-1], x.shape[-1], device=x.device, requires_grad=True)

        Ax = self.A(x)
        Bx = Ax if self.symmetric else self.B(x)
        Ce = self.C(e)

        e = self.propagate(edge_index,
                           Ce=Ce,
                           e=e, Ax=Ax, Bx=Bx)

        return e

    def message(self, Ax_i, Bx_j, Ce):
        e_ij = Ax_i + Bx_j + Ce
        e_ij = torch.nn.functional.relu(self.bn(e_ij))
        e_ij = self.edge_out_proj(e_ij)

        sigma_ij = e_ij
        # sigma_ij = torch.sigmoid(e_ij)
        return sigma_ij

    def aggregate(self, e):
        return e


# returns if 1./0 if (*real*) edge is inside shape
# useage:
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
