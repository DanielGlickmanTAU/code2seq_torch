import unittest
from unittest import TestCase

from torch_geometric.datasets import FakeDataset
import torch

from examples.graphproppred.mol import pygraph_utils
from examples.graphproppred.mol.pygraph_utils import concat_layer_activations
from graphgps.layer.graph_attention.positional.positional_attention_weight import AdjStack

import test_utils


class Test(TestCase):
    def test_get_dense_x_adjstack_mask(self):
        embed_dim = 300
        batch_size = 2
        num_stacks = 4
        args = test_utils.get_args_with_adj_stack(num_stacks)
        dataset = FakeDataset(num_graphs=4, avg_num_nodes=100, num_channels=embed_dim,
                              transform=AdjStack(args))

        first_batch = test_utils.as_pyg_batch(dataset, batch_size)

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
        args = test_utils.get_args_with_adj_stack(num_stacks)

        dataset = FakeDataset(num_graphs=4, avg_num_nodes=100, num_channels=embed_dim,
                              # transform=AdjStack(steps=[1, 2, 3, 4]))
                              )

        first_batch = test_utils.as_pyg_batch(dataset, batch_size)

        prev_x = first_batch.x
        x, mask = pygraph_utils.get_dense_x_and_mask(prev_x, first_batch.batch)
        x_restored = pygraph_utils.get_spare_x(x, ~mask)
        print(x_restored.shape, prev_x.shape)
        self.assertTrue((x_restored == prev_x).all())

    def test_concat_layer_activations(self):
        # 2 nodes with hidden dim 3
        layer1 = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ]
        )
        layer2 = torch.tensor([
            [10, 20, 30],
            [40, 50, 60]
        ]
        )
        result = concat_layer_activations([layer1, layer2], join_dims=True)
        print(result)
        assert (result == torch.tensor(
            torch.tensor([
                [1, 2, 3],
                [10, 20, 30],
                [4, 5, 6],
                [40, 50, 60]
            ]
            )
        )).all()

    def test_to_dense_joined_batch(self):
        embed_dim = 300
        batch_size = 2
        num_stacks = 4
        num_graphs = 4
        dataset = FakeDataset(num_graphs=num_graphs, avg_num_nodes=100, num_channels=embed_dim)
        first_batch = test_utils.as_pyg_batch(dataset)

        layer_one_output = first_batch.x
        layer_two_output = torch.full_like(layer_one_output, 1)
        layer_theree_output = torch.full_like(layer_one_output, 2)

        layers = [layer_one_output, layer_two_output, layer_theree_output]
        layer_history = pygraph_utils.concat_layer_activations(activations=layers)

        x, mask = pygraph_utils.to_dense_joined_batch(layer_history, first_batch.batch,
                                                      joined_graphs=len(layers))

        print(layer_history.shape)
        print(layer_one_output.shape)
        print(x.shape)
        print(x[mask].shape)
        self.assertTrue((x[mask] == layer_history).all())


if __name__ == '__main__':
    # unittest.main()
    Test().test_to_dense_joined_batch()
    # Test().test_concat_layer_activations()
