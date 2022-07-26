import torch
import torch_geometric.transforms
from torch import nn
from torch_geometric.data import Data

# import global_config
# from arg_parse_utils import bool_
from torch_geometric.utils import to_dense_adj


class AdjStackAttentionWeights(torch.nn.Module):
    _stack_dim = 1

    def __init__(self, num_adj_stacks, num_heads, ffn=True, ffn_hidden_multiplier=2):
        super(AdjStackAttentionWeights, self).__init__()
        self.num_adj_stacks = num_adj_stacks
        self.num_heads = num_heads
        if ffn:
            hidden_dim = num_adj_stacks * ffn_hidden_multiplier
            self.weight = torch.nn.Sequential(
                torch.nn.Linear(num_adj_stacks, hidden_dim),
                # torch_g.instance_norm.py
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, num_heads),
            )
        else:
            self.weight = nn.Linear(in_features=num_adj_stacks, out_features=num_heads)

    # stacks shape is (batch,n,n,num_adj_stacks)
    # mask shape is (batch,n,n). True where should hide
    # returns (batch,num_heads,n,n)
    def forward(self, stacks: torch.Tensor, mask=None):
        b, n, n1, num_stacks = stacks.shape
        assert num_stacks == self.num_adj_stacks

        # shape as (batch*n*n, num_stacks)
        # stacks = stacks.permute(0, 2, 3, 1).reshape(-1, self.num_adj_stacks)
        # real_nodes_edge_mask = self._create_real_edges_mask(b, mask, n, stacks)
        # adj_weights = self.weight(stacks[real_nodes_edge_mask])

        adj_weights = self.weight(stacks)
        # assert adj_weights.shape[-1] == self.num_heads

        return adj_weights
        # new_adj = torch.zeros((b * n * n, self.num_heads), device=stacks.device)
        # new_adj[real_nodes_edge_mask] = adj_weights
        # # back to (batch,num_heads,n,n)
        # new_adj = new_adj.view(b, n, n, self.num_heads).permute(0, 3, 1, 2)
        # return new_adj

    def _create_real_edges_mask(self, b, mask, n, stacks):
        # if not global_config.mask_far_away_nodes:
        #     raise Exception('assuming this is true for now')
        #     return stacks.sum(dim=-1) != 0

        if mask is None:
            mask = torch.zeros((b, n, n), device=stacks.device, dtype=torch.bool)
        real_nodes_edge_mask = ~mask.view(-1)
        return real_nodes_edge_mask


def compute_diag(A: torch.Tensor):
    degrees = A.sum(dim=0)
    return torch.diag_embed(degrees)


class AdjStack(torch.nn.Module):

    def __init__(self, steps: list):
        super().__init__()
        self.steps = steps
        assert len(self.steps) == len(set(self.steps)), f'duplicate power in {self.steps}'

    def forward(self, batch, mask):
        # to_dense_adj(batch.edge_index,batch.batch,batch.edge_attr)
        adj = to_dense_adj(batch.edge_index, batch.batch)
        adj = self.to_P_matrix(adj)

        powers = self._calc_power(adj, self.steps)

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
