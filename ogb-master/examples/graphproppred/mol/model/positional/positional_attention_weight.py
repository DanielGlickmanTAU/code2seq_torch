from typing import Union

import torch
import torch_geometric.transforms
from torch import nn
from torch_geometric.data import Data


class AdjStackAttentionWeights(torch.nn.Module):
    _stack_dim = 1

    def __init__(self, num_adj_stacks, num_heads, bias=True):
        super(AdjStackAttentionWeights, self).__init__()
        self.num_adj_stacks = num_adj_stacks
        self.num_heads = num_heads
        self.weight = nn.Linear(in_features=num_adj_stacks, out_features=num_heads, bias=bias)

    # stacks shape is (batch,num_adj_stacks,n,n)
    # mask shape is (batch,n,n). True where should hide
    # returns (batch,num_heads,n,n)
    def forward(self, stacks: torch.Tensor, mask=None):
        b, num_stacks, n, n1, = stacks.shape
        assert num_stacks == self.num_adj_stacks
        if mask is None:
            mask = torch.zeros((b, n, n), device=stacks.device,dtype=torch.bool)
        real_nodes_edge_mask = ~mask.view(-1)
        # shape as (batch*n*n, num_stacks)
        stacks = stacks.permute(0, 2, 3, 1).reshape(-1, self.num_adj_stacks)

        adj_weights = self.weight(stacks[real_nodes_edge_mask])
        new_adj = torch.zeros((b * n * n, self.num_heads), device=stacks.device)

        new_adj[real_nodes_edge_mask] = adj_weights
        # back to (batch,num_heads,n,n)
        new_adj = new_adj.view(b, n, n, self.num_heads).permute(0, 3, 1, 2)
        return new_adj


def compute_diag(A: torch.Tensor):
    degrees = A.sum(dim=0)
    return torch.diag_embed(degrees)


def to_P_matrix(A: torch.Tensor):
    return A / A.sum(dim=-1, keepdim=True)


class AdjStack(torch_geometric.transforms.BaseTransform):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--adj_stacks', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                            help='list of powers to raise and stack the adj matrix.')

    def __init__(self, args: Union[object, list]):
        self.adj_stacks = args if isinstance(args, list) else args.adj_stacks
        assert len(self.adj_stacks) == len(set(self.adj_stacks)), f'duplicate power in {self.adj_stacks}'

    def __call__(self, data: Data):
        edge_index = data.edge_index
        # N = data.x.size(0)
        N = data.num_nodes
        # (row, col) = data.edge_index
        adj = torch.full([N, N], 0)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        adj.fill_diagonal_(0)

        adj = to_P_matrix(adj)
        adj_stack = torch.stack([torch.matrix_power(adj, exp) for exp in self.adj_stacks])
        # need this for now
        adj_stack = adj_stack.numpy()
        data.adj_stack = adj_stack

        return data


def count_paths_cycles(A: torch.Tensor, p):
    D = torch.zeros_like(A)
    C1 = A.clone()
    Cq = C1
    for q in range(p):
        C1Cq, CqC1 = C1 @ Cq, Cq @ C1

        D = torch.diag_embed(C1Cq.diagonal())
        # I think should be here something like  Cq = C1cQ and reduce previous cq
        Cq = C1Cq + CqC1
        Cq.fill_diagonal_(0)
    return Cq, D
