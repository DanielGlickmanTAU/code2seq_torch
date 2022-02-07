import unittest
from unittest import TestCase

from code2seq.utils import compute

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader
import torch

import pygraph_utils
from model.positional.positional_attention_weight import AdjStack
from pygraph_utils import get_dense_x_and_mask


class Test(TestCase):

    def test_get_spare_x(self):
        embed_dim = 300
        batch_size = 2
        num_stacks = 4
        dataset = FakeDataset(num_graphs=4, avg_num_nodes=100, num_channels=embed_dim,
                              transform=AdjStack(list(range(num_stacks))))

        loader = DataLoader(dataset, batch_size=batch_size)

        first_batch = list(loader)[0]

        prev_x = first_batch.x
        x, mask = pygraph_utils.get_dense_x_and_mask(prev_x, first_batch.batch)

        x_restored = pygraph_utils.get_spare_x(x, mask)
        self.assertTrue((x_restored == prev_x).all())


if __name__ == '__main__':
    unittest.main()
