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
        # 1,1 is for broadcasting
        self.weight = nn.Linear(in_features=num_adj_stacks, out_features=num_heads, bias=bias)

    # stacks shape is (batch,num_adj_stacks,n,n)
    # returns (batch,num_heads,n,n)
    def forward(self, stacks: torch.Tensor):
        b, num_stacks, n, n1, = stacks.shape
        assert num_stacks == self.num_adj_stacks
        # shape as (batch*n*n, num_stacks)
        stacks = stacks.permute(0, 2, 3, 1).reshape(-1, self.num_adj_stacks)
        stacks = self.weight(stacks)
        # back to (batch,num_heads,n,n)
        stacks = stacks.view(b, n, n, self.num_heads).permute(0, 3, 1, 2)
        return stacks


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
