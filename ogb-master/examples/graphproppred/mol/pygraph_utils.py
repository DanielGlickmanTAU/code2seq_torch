from typing import Union

import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F

# gets tensor in shape (n,batch_size,d) and num_heads
# returns tensor in shape (batch_size*num_heads, n ,d/num_head)
from torch_scatter import scatter_add


def reshape_to_multihead(tensor: Tensor, num_heads: int):
    n, batch_size, embed_dim = tensor.shape
    head_dim = embed_dim // num_heads
    assert head_dim == embed_dim / num_heads, 'tensor dim must be divisible by num_heads'
    return tensor.contiguous().view(n, batch_size * num_heads, head_dim).transpose(0, 1)


def reshape_attention_mask_to_multihead(attention_mask, num_heads):
    if attention_mask.dim() == 2:
        batch_size, n = attention_mask.shape
        n1 = 1

    elif attention_mask.dim() == 3:
        batch_size, n1, n = attention_mask.shape
        assert n1 == n

    return attention_mask \
        .view(batch_size, 1, n1, n) \
        .expand(-1, num_heads, -1, -1) \
        .reshape(batch_size * num_heads, n1, n)


def compute_batch_usage(batched_data):
    sizes = get_graph_sizes(batched_data)
    used = sum(sizes)
    allocated = max(sizes) * len(sizes)
    return used / allocated


def split_into_graphs(batched_data, h_node):
    graph_end_indexes = get_graph_sizes(batched_data)
    graph_end_indexes_as_list = [x.item() for x in graph_end_indexes]
    h_node_batched = torch.split(h_node, graph_end_indexes_as_list)

    return h_node_batched


# gets a pyG batch of n graphs
# returns a tensor containing the sizes(number of nodes) of each of the graphs
def get_graph_sizes(batching_data: Union[Batch, torch.Tensor]):
    if isinstance(batching_data, torch_geometric.data.Batch):
        batching_data: Tensor = batching_data.batch
    return torch.unique_consecutive(batching_data, return_counts=True)[1]


# B,n,n mask is True for hidden node and False for real node
def get_dense_x_and_mask(x, batch):
    x, node_mask = torch_geometric.utils.to_dense_batch(x, batch)

    return x, attn_mask_to_dense_mask(batch, node_mask)


# gets attn mask of shape (B,N), e.g obtained with torch_geometric.utils.to_dense_batch
# returns (B,N,N) mask. useful e.g for edge wise operation with batch norm
def attn_mask_to_dense_mask(batch, attn_mask):
    batch_size, N = attn_mask.shape

    masks = torch.ones((batch_size, N, N), dtype=torch.bool, device=attn_mask.device)
    for mask, real_size, in zip(masks, get_graph_sizes(batch)):
        mask[0:real_size, 0:real_size] = False
    return ~masks


def dense_mask_to_attn_mask(dense_mask):
    assert dense_mask.dim() == 3
    return dense_mask[:, 0]


# works like torch_geometric.utils.to_dense_batch masking
def _mask(batch) -> Tensor:
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
