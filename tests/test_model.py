import unittest
from unittest import TestCase

import torch
from torch import optim

from code2seq.utils import compute
from args_parse import get_default_args
from data import dataloader_utils
from data.dataloader_utils import get_train_val_test_loaders, transform_to_one_hot
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from train.eval import evaluate
from train.training import train_epoch
from torchvision import transforms


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

        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)
        train_epoch(model, device, train_loader, optimizer, dataset.task_type, assert_no_zero_grad=True)

    def test_can_overfit_molhiv_with_positional_attention(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4

        self.assert_overfit_on_train(args, dataset_samples)

    def test_can_overfit_molhiv_with_content_attention(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'content'
        args.num_layer = args.num_transformer_layers = 4

        self.assert_overfit_on_train(args, dataset_samples)

    def test_can_overfit_molhiv_with_2_gnn_and_content_attention(self):
        dataset_samples = 64
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.attention_type = 'content'
        args.num_layer = 6
        args.num_transformer_layers = 2
        # torch.autograd.set_detect_anomaly(True)
        self.assert_overfit_on_train(args, dataset_samples)
        # torch.autograd.set_detect_anomaly(False)

    def test_can_overfit_molhiv_with_gnn(self):
        dataset_samples = 32 * 10
        # Training settings
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.num_layer = 6
        args.num_transformer_layers = 0
        torch.autograd.set_detect_anomaly(True)
        self.assert_overfit_on_train(args, dataset_samples)
        torch.autograd.set_detect_anomaly(False)

    def assert_overfit_on_train(self, args, dataset_samples, score_needed=0.95):
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.

        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=AdjStack(args))
        device = compute.get_device()
        evaluator = Evaluator(args.dataset)
        train_loader, _, __ = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                         batch_size=args.batch_size,
                                                         limit=dataset_samples)
        model = get_model(args, dataset.num_tasks, device, task='mol')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, dataset.task_type, score_needed)

    def _train_and_assert_overfit_on_train(self, model, train_loader, evaluator, task_type, score_needed=0.9):
        device = compute.get_device()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        for epoch in range(1, 200 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type)
            print(f'loss is {epoch_avg_loss}')

            eval_dict = evaluate(model, device, train_loader, evaluator)
            assert len(eval_dict) == 1
            if 'rocauc' in eval_dict:
                metric = 'rocauc'
            elif 'acc' in eval_dict:
                metric = 'acc'

            rocauc = eval_dict[metric]
            if rocauc > score_needed:
                break
            print(f'Evaluating epoch {epoch}...{metric}: {rocauc}')
        assert rocauc > score_needed

    def test_position_model_dropout_defaults_to_same_as_overall_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[
                   0].dropout.p == model.gnn_transformer.gnn_node.drop_ratio == args.drop_ratio

    def test_position_model_dropout_can_be_different_than_gnn_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2
        args.transformer_encoder_dropout = 0.123

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[0].dropout.p != model.gnn_transformer.gnn_node.drop_ratio

    def test_can_overfit_pattern_dataset_with_position_attention(self):
        # assert False
        dataset_samples = 32 * 100
        # dataset_samples = 2
        # Training settings
        args = get_default_args()
        args.dataset = "PATTERN"
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.num_heads = 1

        device = compute.get_device()

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transforms.Compose(
                                                                                  [transform_to_one_hot, AdjStack(
                                                                                      args)]))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, 1, device, task='pattern')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification')

    def test_can_overfit_pattern_dataset_with_gnn(self):
        dataset_samples = 32 * 100
        args = get_default_args()
        args.dataset = "PATTERN"
        args.num_layer = 2
        args.num_transformer_layers = 0
        args.drop_ratio = 0.
        args.emb_dim = 30
        args.num_heads = 1

        device = compute.get_device()

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transforms.Compose(
                                                                                  [transform_to_one_hot, AdjStack(
                                                                                      args)]))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, 1, device, task='pattern')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification',
                                                score_needed=0.88)

    "comparing with" \
    "https://arxiv.org/pdf/2003.00982.pdf" \
    " https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/SBMs_node_classification/gin_net.py" \
    "https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/configs/SBMs_node_clustering_GIN_PATTERN_500k.json"

    def test_gnn_size_like_benchmarking_gnn_paper(self):
        args = get_default_args()
        args.num_layer = 16
        args.num_transformer_layers = 0
        args.emb_dim = 124
        args.num_heads = 1

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        self.assertTrue(0.9 * 500 <= k_n_params <= 1.1 * 500)

    def test_position_transformer_size_like_4_layer_gnn(self):
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'position'
        args.emb_dim = 60
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        k_params_in_4_layer_gnn = 126.5
        self.assertTrue(0.8 * k_params_in_4_layer_gnn <= k_n_params <= 1.2 * k_params_in_4_layer_gnn)

    def test_content_transformer_size_like_4_layer_gnn(self):
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'content'
        args.emb_dim = 52
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        k_params_in_4_layer_gnn = 126.5
        self.assertTrue(0.8 * k_params_in_4_layer_gnn <= k_n_params <= 1.2 * k_params_in_4_layer_gnn)

    def test_content_transformer_uses_distance_when_enabled(self):
        device = compute.get_device()
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'content'
        args.emb_dim = 52
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        args.num_adj_stacks = 1
        args.use_distance_bias = False
        model = get_model(args, 1, device, task='pattern')
        n_params_without_distance_bias = sum(p.numel() for p in model.parameters())

        args.use_distance_bias = True
        model = get_model(args, 1, device, task='pattern')
        n_params_with_distance_bias = sum(p.numel() for p in model.parameters())

        self.assertGreater(n_params_with_distance_bias, n_params_without_distance_bias)

    def test_gnn_gin_hidden_size_taken_from_param(self):
        args = get_default_args()
        args.num_layer = 16
        args.num_transformer_layers = 0
        args.emb_dim = 124
        args.num_heads = 1

        device = compute.get_device()
        # this is the change from the test above
        args.gin_conv_mlp_hidden_breath = 2
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        self.assertTrue(k_n_params > 1.1 * 500)


if __name__ == '__main__':
    unittest.main()
