import unittest
from unittest import TestCase

from torch_geometric.datasets import FakeDataset
import torch

import pygraph_utils
from model.positional.positional_attention_weight import AdjStack
from tests.test_utils import as_pyg_batch


class Test(TestCase):
    def test_get_dense_x_adjstack_mask(self):
        embed_dim = 300
        batch_size = 2
        num_stacks = 4
        dataset = FakeDataset(num_graphs=4, avg_num_nodes=100, num_channels=embed_dim,
                              transform=AdjStack(list(range(num_stacks))))

        first_batch = as_pyg_batch(dataset, batch_size)

        x, mask = pygraph_utils.get_dense_x_and_mask(first_batch.x, first_batch.batch)

        graph_sizes = pygraph_utils.get_graph_sizes(first_batch)
        largest_graph_size = max(graph_sizes)
        smaller_graph_size = min(graph_sizes)
        largest_graph_index = torch.argmax(graph_sizes).item()
        smaller_graph_index = torch.argmin(graph_sizes).item()
        mask_of_larger_graph = mask[largest_graph_index]
        mask_of_smaller_graph = mask[smaller_graph_index]

        self.assertTrue(x.shape == (batch_size, largest_graph_size, embed_dim))
        self.assertTrue(mask.shape == (batch_size, largest_graph_size, largest_graph_size))

        self.assertFalse(mask_of_larger_graph.any(), msg='should not mask anything in largest graph')
        self.assertFalse(mask_of_smaller_graph[:smaller_graph_size, :smaller_graph_size].any(),
                         msg='should not hide in the real node parts')
        self.assertTrue(mask_of_smaller_graph[smaller_graph_size:].all(),
                        msg="should hide all fake nodes")

        adj_stack = pygraph_utils.get_dense_adjstack(first_batch.adj_stack, first_batch.batch)
        self.assertTrue(adj_stack.shape == (batch_size, num_stacks, largest_graph_size, largest_graph_size))
        # we want adj_stack in the places of real nodes(by mask) to be none zero, and zero where mask is true
        for stack_dim in adj_stack.transpose(0, 1):
            self.assertFalse(stack_dim[mask.bool()].any())

    def test_reshape_attention_mask_to_multihead(self):
        batch_size = 32
        n = 50
        num_heads = 4
        mask = torch.randint(0, 1 + 1, (batch_size, n, n)).bool()

        # bsz * num_heads, tgt_len, src_len
        shaped_mask = pygraph_utils.reshape_attention_mask_to_multihead(mask, num_heads)
        self.assertEqual(shaped_mask.shape, (batch_size * num_heads, n, n))
        batch1_head1_mask, batch1_head2_mask, batch1_head3_mask, batch1_head4_mask = shaped_mask[:num_heads]
        batch2_head1_mask = shaped_mask[num_heads]

        self.assertTrue((batch1_head1_mask == batch1_head2_mask).all())
        self.assertTrue((batch1_head2_mask == batch1_head3_mask).all())
        self.assertTrue((batch1_head3_mask == batch1_head4_mask).all())

        self.assertFalse((batch1_head1_mask == batch2_head1_mask).all())

    def test_get_spare_x(self):
        embed_dim = 300
        batch_size = 2
        num_stacks = 4
        dataset = FakeDataset(num_graphs=4, avg_num_nodes=100, num_channels=embed_dim,
                              transform=AdjStack(list(range(num_stacks))))

        first_batch = as_pyg_batch(dataset, batch_size)

        prev_x = first_batch.x
        x, mask = pygraph_utils.get_dense_x_and_mask(prev_x, first_batch.batch)

        x_restored = pygraph_utils.get_spare_x(x, mask)
        self.assertTrue((x_restored == prev_x).all())


if __name__ == '__main__':
    unittest.main()
