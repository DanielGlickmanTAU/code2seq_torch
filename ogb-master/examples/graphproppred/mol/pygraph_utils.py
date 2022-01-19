from typing import Union

import torch
import torch_geometric
from commode_utils.training import cut_into_segments
from torch import Tensor
from torch_geometric.data import Batch


def split_into_graphs(batched_data, h_node):
    graph_end_indexes = get_graph_sizes(batched_data)
    graph_end_indexes_as_list = [x.item() for x in graph_end_indexes]
    h_node_batched = torch.split(h_node, graph_end_indexes_as_list)

    return h_node_batched


#gets pyg data and
def batch_tensor_by_pyg_indexes(tensor: Tensor, batching_data: Union[Batch, torch.Tensor]):
    batched_contexts, attention_mask = cut_into_segments(tensor, get_graph_sizes(batching_data))
    return batched_contexts, attention_mask


# gets a pyG batch of n graphs
# returns a tensor containing the sizes(number of nodes) of each of the graphs
def get_graph_sizes(batching_data: Union[Batch, torch.Tensor]):
    if isinstance(batching_data, torch_geometric.data.Batch):
        batching_data: Tensor = batching_data.batch
    return torch.unique_consecutive(batching_data, return_counts=True)[1]
