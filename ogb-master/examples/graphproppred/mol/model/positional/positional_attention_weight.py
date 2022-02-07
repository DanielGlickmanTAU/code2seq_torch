from typing import Union

import torch
import torch_geometric.transforms
from torch_geometric.data import Data


class AdjStackAttentionWeights(torch.nn.Module):
    _stack_dim = 1

    def __init__(self, num_adj_stacks, num_heads, num_conv=None):
        super(AdjStackAttentionWeights, self).__init__()
        assert num_conv is None  # in progress
        num_conv = num_adj_stacks
        self.num_adj_stacks = num_adj_stacks
        self.num_heads = num_heads

    def forward(self, stacks: torch.Tensor):
        stacks_ = stacks.float().mean(dim=1)
        return stacks_.unsqueeze(1).repeat(1, self.num_heads, 1, 1)





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

        adj_stack = torch.stack([torch.matrix_power(adj, exp) for exp in self.adj_stacks])
        # need this for now
        adj_stack = adj_stack.numpy()
        data.adj_stack = adj_stack

        return data
