from typing import Union

import torch
import torch_geometric
from commode_utils.training import cut_into_segments
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F


# gets tensor in shape (n,batch_size,d) and num_heads
# returns tensor in shape (batch_size*num_heads, n ,d/num_head)
def reshape_to_multihead(tensor: Tensor, num_heads: int):
    n, batch_size, embed_dim = tensor.shape
    head_dim = embed_dim // num_heads
    assert head_dim == embed_dim / num_heads, 'tensor dim must be divisible by num_heads'
    return tensor.contiguous().view(n, batch_size * num_heads, head_dim).transpose(0, 1)


def reshape_attention_mask_to_multihead(attention_mask, num_heads):
    batch_size, n, n2 = attention_mask.shape
    assert n == n2

    contiguous_mask = attention_mask.unsqueeze(1).contiguous()
    mask_duplicated_for_each_head_in_same_batch = contiguous_mask.repeat(1, num_heads, 1, 1)
    return mask_duplicated_for_each_head_in_same_batch.view(batch_size * num_heads, n, n)


def split_into_graphs(batched_data, h_node):
    graph_end_indexes = get_graph_sizes(batched_data)
    graph_end_indexes_as_list = [x.item() for x in graph_end_indexes]
    h_node_batched = torch.split(h_node, graph_end_indexes_as_list)

    return h_node_batched


# gets pyg data and
def batch_tensor_by_pyg_indexes(tensor: Tensor, batching_data: Union[Batch, torch.Tensor]):
    batched_contexts, attention_mask = cut_into_segments(tensor, get_graph_sizes(batching_data))
    return batched_contexts, attention_mask


# gets a pyG batch of n graphs
# returns a tensor containing the sizes(number of nodes) of each of the graphs
def get_graph_sizes(batching_data: Union[Batch, torch.Tensor]):
    if isinstance(batching_data, torch_geometric.data.Batch):
        batching_data: Tensor = batching_data.batch
    return torch.unique_consecutive(batching_data, return_counts=True)[1]


# mask is True for hidden node and False for real node
def get_dense_x_and_mask(x, batch):
    x, node_mask = torch_geometric.utils.to_dense_batch(x, batch)
    batch_size, N = node_mask.shape

    masks = torch.ones((batch_size, N, N), dtype=torch.bool)
    for mask, real_size, in zip(masks, get_graph_sizes(batch)):
        mask[0:real_size, 0:real_size] = False
    return x, masks


def get_spare_x(dense_x, origin_mask):
    assert origin_mask.dim() == 3
    # take attention mask of shape (b,N,N) and extracting just the node mask
    node_mask = origin_mask[:, 0]
    real_nodes_indicator = ~node_mask
    real_nodes_index = real_nodes_indicator.nonzero()
    batch_index, node_index = real_nodes_index[:, 0], real_nodes_index[:, 1]

    real_nodes_selected = dense_x[batch_index, node_index]
    return real_nodes_selected


def get_dense_adjstack(stack_list: list, batch):
    largest_graph_size = max(get_graph_sizes(batch))
    padded_stacks = []
    for stack in stack_list:
        stack = torch.tensor(stack)
        graph_size = stack.shape[-1]
        padding = largest_graph_size - graph_size
        padding_left, padding_right, padding_top, padding_bottom = 0, padding, 0, padding
        padded_stack = F.pad(stack, (padding_left, padding_right, padding_top, padding_bottom))
        padded_stacks.append(padded_stack)
    return torch.stack(padded_stacks).to(batch.device)