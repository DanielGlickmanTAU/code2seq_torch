import torch
from torch import nn, Tensor
from torch_scatter import scatter_add

# import global_config
# from arg_parse_utils import bool_
from torch_geometric.utils import to_dense_adj, to_dense_batch
from typing import Optional

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

        adj_weights = torch.zeros((b, n1, n, self.num_heads), device=stacks.device)
        adj_weights[mask] = self.weight(stacks[mask].view(-1, num_stacks)).view(-1, n, self.num_heads)
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


class Diffuser(nn.Module):
    def __init__(self, nagasaki_config):
        super().__init__()
        steps = nagasaki_config.steps
        if isinstance(steps, str):
            steps = list(eval(steps))
        num_stack = positional_utils.get_stacks_dim(nagasaki_config)
        edge_dim = positional_utils.get_edge_dim(nagasaki_config)

        self.adj_stacker = AdjStack(steps)

        self.edge_mlp = AdjStackAttentionWeights(
            input_dim=num_stack,
            hidden_dim=edge_dim,
            dim_out=edge_dim,
            ffn=nagasaki_config.edge_model_type,
            ffn_layers=nagasaki_config.ffn_layers)

    def forward(self, batch):
        mask = self._mask(batch.batch)
        stacks = self.adj_stacker(batch, mask)
        edges = self.edge_mlp(stacks, mask)
        batch.edges = edges
        return batch

    def _mask(self, batch) -> Tensor:
        batch_size = int(batch.max()) + 1

        num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0,
                                dim_size=batch_size)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

        max_num_nodes = int(num_nodes.max())

        idx = torch.arange(batch.size(0), dtype=torch.long, device=batch.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

        mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                           device=batch.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)

        return mask
