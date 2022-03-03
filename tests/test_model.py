import unittest
from unittest import TestCase

import torch
from torch import optim

import consts
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
    def test_can_overfit_molhiv_with_positional_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'position'

        self.assert_overfit_on_train(args, dataset_samples)

    def test_can_overfit_molhiv_with_content_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'content'

        self.assert_overfit_on_train(args, dataset_samples)

    def test_can_overfit_molhiv_with_2_gnn_and_content_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'content'
        args.num_layer = 6
        args.num_transformer_layers = 2
        self.assert_overfit_on_train(args, dataset_samples)

    def test_can_overfit_molhiv_with_gnn(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.num_transformer_layers = 0
        self.assert_overfit_on_train(args, dataset_samples)

    def _get_molhiv_overfit_params(self):
        dataset_samples = 64
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.num_layer = args.num_transformer_layers = 4
        return args, dataset_samples

    # works only on molhiv
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
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(1, 200 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type)
            print(f'loss is {epoch_avg_loss}')

            eval_dict = evaluate(model, device, train_loader, evaluator)
            if 'rocauc' in eval_dict:
                metric = 'rocauc'
            elif 'acc' in eval_dict:
                metric = 'acc'

            rocauc = eval_dict[metric]
            if rocauc > score_needed:
                break
            print(f'Evaluating epoch {epoch}...{metric}: {eval_dict}')
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
        model = get_model(args, consts.pattern_num_tasks, device, task='pattern')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification')

    def test_can_overfit_pattern_dataset_with_content_attention_and_distance(self):
        dataset_samples = 32 * 4
        args = get_default_args()
        args.dataset = "PATTERN"
        args.attention_type = 'content'
        args.use_distance_bias = True
        args.num_layer = args.num_transformer_layers = 1
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
        model = get_model(args, consts.pattern_num_tasks, device, task='pattern')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification')

    def test_can_overfit_pattern_dataset_with_gnn(self):
        dataset_samples = 32 * 4
        args = get_default_args()
        # dataset_samples = 2
        args.emb_dim = 30
        args.dataset = "PATTERN"
        args.num_layer = 4
        args.num_transformer_layers = 0
        args.drop_ratio = 0.
        args.batch_size = 128
        args.JK = 'sum'
        # args.JK = 'last'

        device = compute.get_device()

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transforms.Compose(
                                                                                  [transform_to_one_hot, AdjStack(
                                                                                      args)]))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, device, task='pattern')
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification',
                                                score_needed=0.88)


if __name__ == '__main__':
    unittest.main()
