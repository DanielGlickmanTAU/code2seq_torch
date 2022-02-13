from code2seq.utils import compute
import argparse
import unittest
from unittest import TestCase

import torch
from torch import optim

from args_parse import get_default_args

from data import dataloader_utils
from data.dataloader_utils import get_train_val_test_loaders
from model.ContentMultiHeadAttention import ContentMultiheadAttention
from model.model_utils import get_model
from model.positional.PositionMultiHeadAttention import PositionMultiHeadAttention
from model.positional.positional_attention_weight import AdjStack, AdjStackAttentionWeights
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from train import train_epoch, evaluate


class Test(TestCase):

    def test_grads_and_outputs_content_attention(self):
        # Training settings
        args = get_default_args()
        args.attention_type = 'content'
        self._assert_output_and_grad(args)

    def test_grads_and_outputs_position_attention(self):
        def callback(model):
            def position_attention_backwards_hook(module, grad_input, grad_output):
                assert not grad_output[0].isnan().any()
                model.gnn_transformer.transformer.layers[
                    0].attention_layer.positional_bias.weight.register_full_backward_hook(
                    position_attention_backwards_hook)

        # Training settings
        args = get_default_args()
        args.attention_type = 'position'
        self._assert_output_and_grad(args, callback)

    @staticmethod
    def assert_not_nan_or_inf(input_tensor):
        if isinstance(input_tensor, tuple) and input_tensor:
            input_tensor = input_tensor[0]
            if isinstance(input_tensor, torch.Tensor):
                assert not input_tensor.isnan().any()
                assert not input_tensor.isinf().any()

    def _assert_output_and_grad(self, args, model_callback=None):
        dataset_samples = 64
        args.dataset = "ogbg-molhiv"
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
            self.assert_not_nan_or_inf(input)
            self.assert_not_nan_or_inf(output)

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]

            if grad_input and grad_input[0] is not None:
                if isinstance(grad_input, tuple):
                    grad_input = grad_input[0]
                assert not grad_input.isnan().any()

            # assert not grad_output.isnan().any()

        torch.nn.modules.module.register_module_forward_hook(hook)

        torch.nn.modules.module.register_module_full_backward_hook(backward_hook)
        if model_callback:
            model_callback(model)

        train_epoch(model, device, train_loader, optimizer, dataset.task_type)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type)

    def test_can_overfit_molhiv_with_positional_attention(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.

        self.assert_overfit(args, dataset_samples)

    def test_can_overfit_molhiv_with_content_attention(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'content'
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.

        self.assert_overfit(args, dataset_samples)

    def assert_overfit(self, args, dataset_samples):
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
        for epoch in range(1, 10):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, "classification")

            print(f'Evaluating epoch {epoch}')
            print(evaluate(model, device, test_loader, evaluator))


if __name__ == '__main__':
    unittest.main()
