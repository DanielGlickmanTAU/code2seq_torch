import torch
from torch import nn
from examples.graphproppred.mol.pygraph_utils import get_dense_x_and_mask

from graphgps.layer.graph_attention.positional import positional_utils


class CLSNode(nn.Module):
    def __init__(self, dim_in, nagasaki_config):
        super().__init__()
        self.edge_dim = positional_utils.get_edge_dim(nagasaki_config)
        self.cls = nn.Parameter(torch.randn([1, 1, dim_in], requires_grad=True))
        self.cls_edge = nn.Parameter(torch.randn([1, 1, 1, self.edge_dim], requires_grad=True))

    def forward(self, batch):
        x, mask = get_dense_x_and_mask(batch.x, batch.batch)
        edges = batch.edges

        num_graphs = x.size(0)
        # cls_batch = self.cls.to(x).expand(num_graphs, -1, -1)
        # x = torch.cat([x, cls_batch], dim=1)
        cls_edge_batch = self.cls_edge.to(x).expand(num_graphs, -1, -1, -1)
        # make all existing real nodes look at cls.
        real_nodes_mask = mask[:, 0]
        mask = torch.cat([mask, real_nodes_mask.unsqueeze(2)], dim=2)
        # make cls look at all existing real nodes
        real_nodes_mask = mask[:, 0]
        mask = torch.cat([mask, real_nodes_mask.unsqueeze(1)], dim=1)

        edges = torch.cat([edges, cls_edge_batch.expand(-1, -1, edges.size(2), -1)], dim=1)
        edges = torch.cat([edges, cls_edge_batch.expand(-1, edges.size(1), -1, -1)], dim=2)

        cls = self.cls.to(batch.x).squeeze()
        new_x = []
        cls_mask = [False] * (len(batch.x) + num_graphs)
        graph_starts = batch.ptr

        for i in range(len(graph_starts) - 1):
            start_node = graph_starts[i]
            end_node = graph_starts[i + 1]
            new_x += batch.x[start_node:end_node]
            new_x.append(cls)
            cls_mask[(len(new_x)) - 1] = True

        batch.cls_mask = torch.tensor(cls_mask, device=batch.x.device)
        batch.x = torch.stack(new_x)
        batch.batch = torch.sort(
            torch.cat([batch.batch, torch.arange(0, batch.num_graphs, device=batch.x.device)])).values
        batch.ptr = batch.ptr + torch.arange(0, batch.num_graphs + 1, device=batch.x.device)
        batch.mask = mask
        batch.edges = edges

        return batch


class CLSHead(nn.Module):
    def __init__(self, dim_in, dim_out, task):
        super(CLSHead, self).__init__()
        #only graph prediction is supported with cls right now
        assert task in {'graph'}
        self.task = task
        self.pred_linear = nn.Linear(dim_in, dim_out)

    def forward(self, batch):
        if self.task == 'graph':
            x = batch.x[batch.cls_mask]
        else:  # node prediction
            x = batch.x[~batch.cls_mask]
        return self.pred_linear(x), batch.y
