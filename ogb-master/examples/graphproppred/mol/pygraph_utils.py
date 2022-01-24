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


"from https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py"
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
