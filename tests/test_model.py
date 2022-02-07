import argparse
import unittest
from unittest import TestCase

from torch import optim

from args_parse import get_default_args
from code2seq.utils import compute

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader

from data.dataloader_utils import get_train_val_test_loaders
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from train import train_epoch, evaluate


class Test(TestCase):

    def test_get_spare_x(self):
        # Training settings
        parser = get_default_args()
        parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                            help='dataset name (default: ogbg-molhiv)')
        args = parser.parse_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2

        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        device = compute.get_device()

        evaluator = Evaluator(args.dataset)

        train_loader, valid_loader, test_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                             batch_size=args.batch_size, limit=100)
        model = get_model(args, dataset.num_tasks, device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, dataset.task_type)

            print('Evaluating...')
            # train_perf = evaluate(model, device, train_loader, evaluator)
            print(evaluate(model, device, valid_loader, evaluator))


if __name__ == '__main__':
    unittest.main()
