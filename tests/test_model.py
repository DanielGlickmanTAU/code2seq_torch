from code2seq.utils import compute
import argparse
import unittest
from unittest import TestCase

import torch
from torch import optim

from args_parse import get_default_args

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

    def test_can_overfit_molhiv(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.

        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        device = compute.get_device()

        evaluator = Evaluator(args.dataset)

        train_loader, valid_loader, test_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                             batch_size=args.batch_size,
                                                                             limit=dataset_samples)
        model = get_model(args, dataset.num_tasks, device)

        optimizer = optim.Adam(model.parameters(), lr=3e-5)

        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            assert not output.isnan().any()

        torch.nn.modules.module.register_module_forward_hook(hook)

        for epoch in range(1, 100 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, dataset.task_type)

            print(f'Evaluating epoch {epoch}')
            rocauc = evaluate(model, device, test_loader, evaluator)['rocauc']
            print(rocauc)
        assert rocauc > 0.95

    def test_content_position_model_dropout_defaults_to_same_as_overall_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2

        model = get_model(args, 3, compute.get_device())
        assert model.gnn_transformer.transformer.layers[
                   0].dropout.p == model.gnn_transformer.gnn_node.drop_ratio == args.drop_ratio

    def test_content_position_model_can_be_different_than_gnn_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2
        args.transformer_encoder_dropout = 0.123

        model = get_model(args, 3, compute.get_device())
        assert model.gnn_transformer.transformer.layers[0].dropout.p != model.gnn_transformer.gnn_node.drop_ratio



if __name__ == '__main__':
    unittest.main()
