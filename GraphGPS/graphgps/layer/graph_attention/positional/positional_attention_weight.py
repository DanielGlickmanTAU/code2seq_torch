import torch
import torch_geometric.transforms
from torch import nn
from torch_geometric.data import Data

# import global_config
# from arg_parse_utils import bool_
from torch_geometric.utils import to_dense_adj


class AdjStackAttentionWeights(torch.nn.Module):
    _stack_dim = 1

    def __init__(self, num_adj_stacks, num_heads, ffn=True, ffn_hidden_multiplier=2, ffn_layers=1):
        super(AdjStackAttentionWeights, self).__init__()
        self.num_adj_stacks = num_adj_stacks
        self.num_heads = num_heads
        if ffn == 'bn-linear':
            self.weight = torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_adj_stacks),
                torch.nn.Linear(num_adj_stacks, num_heads),

            )
        elif ffn == 'bn-mlp' or ffn == 'mlp':
            hidden_dim = num_adj_stacks * ffn_hidden_multiplier
            if ffn == 'mlp':
                layers = [torch.nn.Linear(num_adj_stacks, hidden_dim), torch.nn.BatchNorm1d(hidden_dim)]
            else:
                layers = [torch.nn.BatchNorm1d(num_adj_stacks), torch.nn.Linear(num_adj_stacks, hidden_dim)]
            layers.append(torch.nn.ReLU())
            for _ in range(ffn_layers - 1):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.Linear(hidden_dim, num_heads))

            self.weight = torch.nn.Sequential(*layers)

        elif ffn == 'linear':
            self.weight = nn.Linear(in_features=num_adj_stacks, out_features=num_heads)
        else:
            raise ValueError(f'nagasaki does not support edge forward model of type {ffn}')

    # stacks shape is (batch,n,n,num_adj_stacks)
    # mask shape is (batch,n).
    # returns (batch,n,n,num_heads)
    def forward(self, stacks: torch.Tensor, mask=None):
        b, n, n1, num_stacks = stacks.shape
        assert num_stacks == self.num_adj_stacks

        # adj_weights = self.weight(stacks)
        # adj_weights = self.weight(stacks.view(-1, num_stacks)).view((b, n1, n, self.num_heads))

        adj_weights = torch.zeros((b, n1, n, self.num_heads), device=stacks.device)
        adj_weights[~mask] = self.weight(stacks[~mask].view(-1, num_stacks)).view(-1, n, self.num_heads)
        assert adj_weights.shape == (b, n, n1, self.num_heads)
        return adj_weights


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
