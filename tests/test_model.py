from code2seq.utils import compute
import argparse
import unittest
from unittest import TestCase

import torch
from torch import optim

from args_parse import get_default_args

from data import dataloader_utils
from data.dataloader_utils import get_train_val_test_loaders
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from train import train_epoch, evaluate


class Test(TestCase):

    def test_grads_and_outputs(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.

        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        device = compute.get_device()

        evaluator = Evaluator(args.dataset)

        train_loader, valid_loader, test_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                             batch_size=args.batch_size,
                                                                             limit=dataset_samples)
        model = get_model(args, dataset.num_tasks, device, task='mol')

        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            isnan__any = output.isnan().any()
            assert not isnan__any

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]

            if grad_input and grad_input[0] is not None:
                if isinstance(grad_input, tuple):
                    grad_input = grad_input[0]
                assert not grad_input.isnan().any()

            # assert not grad_output.isnan().any()

        def position_attention_backwards_hook(module, grad_input, grad_output):
            assert not grad_output[0].isnan().any()

        torch.nn.modules.module.register_module_forward_hook(hook)
        torch.nn.modules.module.register_module_full_backward_hook(backward_hook)
        model.gnn_transformer.transformer.layers[0].attention_layer.positional_bias.weight.register_full_backward_hook(
            position_attention_backwards_hook)

        train_epoch(model, device, train_loader, optimizer, dataset.task_type)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type)

    def test_can_overfit_molhiv(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.

        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        device = compute.get_device()

        evaluator = Evaluator(args.dataset)

        train_loader, valid_loader, test_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                             batch_size=args.batch_size,
                                                                             limit=dataset_samples)
        model = get_model(args, dataset.num_tasks, device, task='mol')

        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        for epoch in range(1, 100 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, dataset.task_type)

            print(f'Evaluating epoch {epoch}')
            rocauc = evaluate(model, device, test_loader, evaluator)['rocauc']
            if rocauc > 0.95:
                break
            print(rocauc)
        assert rocauc > 0.95

    def test_content_position_model_dropout_defaults_to_same_as_overall_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[
                   0].dropout.p == model.gnn_transformer.gnn_node.drop_ratio == args.drop_ratio

    def test_content_position_model_dropout_can_be_different_than_gnn_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2
        args.transformer_encoder_dropout = 0.123

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[0].dropout.p != model.gnn_transformer.gnn_node.drop_ratio

    def test_pattern_dataset(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "PATTERN"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.emb_dim = 3
        args.num_heads = 1

        # dataset = PygGraphPropPredDataset(name=args.dataset,
        #                                   transform=AdjStack(args))

        device = compute.get_device()
        from torchvision import transforms
        def to_int(data):
            data.x = data.x.int()
            return data

        train_loader, valid_loader, test_loader = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                                                  num_workers=args.num_workers,
                                                                                                  batch_size=args.batch_size,
                                                                                                  limit=dataset_samples,
                                                                                                  transform=transforms.Compose(
                                                                                                      [to_int, AdjStack(
                                                                                                          args)]))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, 1, device, task='pattern')

        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        for epoch in range(1, 100 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, "classification")

            print(f'Evaluating epoch {epoch}')
            print(evaluate(model, device, test_loader, evaluator))


if __name__ == '__main__':
    unittest.main()
