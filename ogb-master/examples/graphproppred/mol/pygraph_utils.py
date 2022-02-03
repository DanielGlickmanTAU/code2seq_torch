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
    assert head_dim == embed_dim / num_heads, 'tensor dim must be divisble by num_heads'
    return tensor.contiguous().view(n, batch_size * num_heads, head_dim).transpose(0, 1)


def reshape_attention_mask_to_multihead(attention_mask, num_heads):
    batch_size, n, n2 = attention_mask.shape
    assert n == n2

    return attention_mask.unsqueeze(1).contiguous().repeat(1, 4, 1, 1).view(batch_size * num_heads, n, n)


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


"mask is True for hidden node and False for real node"


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


# dense_x[i, 0:num_node] (106,300)
# mask (122)
# prev_h_node (228,300)
def get_dense_adjstack(stack_list: list, batch):
    # adjstack batch_size, (num_stacks,n,n)
    batch_size = len(stack_list)

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


# "from https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py"
def pad_batch(h_node, batch, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # logger.info(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


def unpad_batch(padded_h_node, prev_h_node, num_nodes, origin_mask, max_num_nodes):
    """
    padded_h_node: [s, b, f]
    prev_h_node: [bxs, f]
    batch: [n]
    pad_mask: [b, s]
    """

    for i, mask in enumerate(origin_mask):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
            # cutoff mask
            indices = mask.nonzero()
            indices = indices[-num_node:]
            mask = torch.zeros_like(mask)
            mask[indices] = True
        # logger.info("prev_h_node:", prev_h_node.size())
        # logger.info("padded_h_node:", padded_h_node.size())
        # logger.info("mask:", mask.size())
        prev_h_node = prev_h_node.masked_scatter(mask.unsqueeze(-1), padded_h_node[-num_node:, i])
    return prev_h_node
