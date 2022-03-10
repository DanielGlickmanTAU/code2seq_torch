import unittest
from unittest import TestCase

import torch
from torch import optim

from model.positional.AttentionWeightNormalizer import AttentionWeightNormalizer
from tests import test_utils
from train.training import train_epoch

from code2seq.utils import compute
from args_parse import get_default_args
from data.dataloader_utils import get_train_val_test_loaders
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import PygGraphPropPredDataset


class Test(TestCase):
    def test_gnn_transformer_shaping(self):
        # Training settings
        args = get_default_args()
        args.attention_type = 'position'

        model = get_model(args, 2, compute.get_device(), task='mol')
        gnn_transformer = model.gnn_transformer
        gnn_transformer.transformer = test_utils.MockModule(lambda h_node_batch, mask, adj_stack: h_node_batch)

        args.dataset = "ogbg-molhiv"
        args.num_layer = args.num_transformer_layers = 1,
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        batch = test_utils.as_pyg_batch(dataset)
        transformed = gnn_transformer.forward_transformer(batch, batch.x)
        self.assertTrue((batch.x == transformed).all())

    def test_grads_and_outputs_content_attention(self):
        # Training settings
        args = get_default_args()
        args.attention_type = 'content'
        self._assert_output_and_grad(args)

    def test_grads_and_outputs_content_attention_with_distance(self):
        # Training settings
        args = get_default_args()
        args.attention_type = 'content'
        args.use_distance_bias = True
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
        train_loader, valid_loader, test_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                             batch_size=args.batch_size,
                                                                             limit=dataset_samples)
        model = get_model(args, dataset.num_tasks, device, task='mol')
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        def hook(model, input, output):
            if type(model) == torch.nn.Softmax:
                return
            self.assert_not_nan_or_inf(input)
            self.assert_not_nan_or_inf(output)

        def backward_hook(module, grad_input, grad_output):
            if type(module) == torch.nn.Softmax:
                return

            if grad_input and grad_input[0] is not None:
                # AttentionWeightNormalizer masks input so I guess its ok to have nan grads...
                if isinstance(grad_input, tuple) and not type(module) == AttentionWeightNormalizer:
                    grad_input = grad_input[0]
                    assert not grad_input.isnan().any()

                if isinstance(grad_output, tuple):
                    grad_output = grad_output[0]
                    assert not grad_output.isnan().any()

        torch.nn.modules.module.register_module_forward_hook(hook)

        torch.nn.modules.module.register_module_full_backward_hook(backward_hook)
        if model_callback:
            model_callback(model)

        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)


if __name__ == '__main__':
    unittest.main()
