import global_config
from code2seq.utils import compute
import unittest
from unittest import TestCase

import torch
from torch import optim, nn

import exp_utils
from tests import test_utils
from train.eval import evaluate
from train.training import train_epoch

import consts
from args_parse import get_default_args
from data import dataloader_utils
from data.dataloader_utils import get_train_val_test_loaders, transform_to_one_hot
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset


class Test(TestCase):
    def test_can_overfit_molhiv_with_positional_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'position'

        self.assert_overfit_on_train_molhiv(args, dataset_samples)

    def test_can_overfit_molhiv_with_content_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'content'

        self.assert_overfit_on_train_molhiv(args, dataset_samples)

    def test_can_overfit_molhiv_with_2_gnn_and_content_attention(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.attention_type = 'content'
        args.num_layer = 6
        args.num_transformer_layers = 2
        self.assert_overfit_on_train_molhiv(args, dataset_samples)

    def test_can_overfit_molhiv_with_gnn(self):
        args, dataset_samples = self._get_molhiv_overfit_params()
        args.num_transformer_layers = 0
        self.assert_overfit_on_train_molhiv(args, dataset_samples)

    def test_can_overfit_pattern_dataset_with_position_batch_norm_both(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = True

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1]
        args.num_heads = 1
        args.learning_rate = 3e-5
        args.attention_norm_type = 'batch'
        args.ff_norm_type = 'batch'

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_gating_ff_batch_norm(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = True

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        args.ff_norm_type = 'batch'

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_gating_large_learning_attention_batch_norm(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = True

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1]
        args.num_heads = 1
        args.learning_rate = 3e-5
        args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_gating__batch_norm_in_mlp(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = True

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position__batch_norm_in_mlp_larger_fov(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3, 4]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = True

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        # exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position__batch_norm_in_mlp(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = True

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        # exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_nobn(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = True

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_gating(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = True

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_nobn3(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3, 4]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = False

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_nobn4(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = False

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3, 4]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'
        args.use_batch_norm_in_transformer_mlp = False

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    def test_can_overfit_pattern_dataset_with_position_gating__batch_norm_in_mlp_big_receptive(self):
        args, dataset_samples = self._get_pattern_overfit_config()

        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 4
        args.gating = True

        args.norm_first = True
        # args.use_distance_bias = True
        args.adj_stacks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        args.num_heads = 1
        args.learning_rate = 3e-5
        # args.attention_norm_type = 'batch'
        # args.ff_norm_type = 'batch'

        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))

        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')

        exp = None
        exp = exp_utils.start_exp("test", args, model)
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp,
                                                lr=args.learning_rate)

    # def test_can_overfit_pattern_dataset_with_position_attention(self):
    #     args, dataset_samples = self._get_pattern_overfit_config()
    #
    #     args.attention_type = 'position'
    #
    #     self.assert_overfit_on_train_pattern(args, dataset_samples)

    def assert_overfit_on_train_pattern(self, args, dataset_samples, exp=False):
        train_loader, _, __ = dataloader_utils.pyg_get_train_val_test_loaders(args.dataset,
                                                                              num_workers=args.num_workers,
                                                                              batch_size=args.batch_size,
                                                                              limit=dataset_samples,
                                                                              transform=transform_to_one_hot,
                                                                              mapping=AdjStack(args))
        evaluator = Evaluator(args.dataset)
        model = get_model(args, consts.pattern_num_tasks, compute.get_device(), task='pattern')
        exp = exp_utils.start_exp("test", args, model) if exp else None
        self._train_and_assert_overfit_on_train(model, train_loader, evaluator, 'node classification', exp=exp)

    # def test_can_overfit_pattern_dataset_with_content_attention_and_distance(self):
    #     args, dataset_samples = self._get_pattern_overfit_config()
    #
    #     args.use_distance_bias = True
    #     args.attention_type = 'content'
    #
    #     self.assert_overfit_on_train_pattern(args, dataset_samples)

    def test_can_overfit_pattern_dataset_with_gnn(self):
        args, dataset_samples = self._get_pattern_overfit_config()
        args.JK = 'sum'
        args.JK = 'last'
        args.num_transformer_layers = 0

        self.assert_overfit_on_train_pattern(args, dataset_samples, exp=False)

    def test_can_overfit_pattern_dataset_with_gnn_TEMP(self):
        args, dataset_samples = self._get_pattern_overfit_config()
        args.JK = 'last'
        args.num_transformer_layers = 0
        args.num_layer = 2

        self.assert_overfit_on_train_pattern(args, dataset_samples, exp=True)

    def test_can_overfit_pattern_dataset_with_gnn_TEMP_no_relu(self):
        args, dataset_samples = self._get_pattern_overfit_config()
        args.JK = 'last'
        args.num_transformer_layers = 0
        args.num_layer = 2
        print(args.mask_far_away_nodes)

        self.assert_overfit_on_train_pattern(args, dataset_samples, exp=True)

    def _get_pattern_overfit_config(self):
        dataset_samples = 32
        args = get_default_args()
        args.dataset = "PATTERN"
        args.num_layer = args.num_transformer_layers = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.num_heads = 1
        return args, dataset_samples

    def _get_molhiv_overfit_params(self):
        dataset_samples = 64
        args = get_default_args()
        args.dataset = "ogbg-molhiv"
        args.num_layer = args.num_transformer_layers = 4
        return args, dataset_samples

    def assert_overfit_on_train_molhiv(self, args, dataset_samples, score_needed=0.95):
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

    def _train_and_assert_overfit_on_train(self, model, train_loader, evaluator, task_type, score_needed=0.9, exp=None,
                                           lr=3e-5):
        print('learning rate and epochjs changed')
        device = compute.get_device()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 500 + 1):
            epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type, experiment=exp)
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
            if exp:
                exp.log_metric('score', rocauc)
        assert rocauc > score_needed


if __name__ == '__main__':
    unittest.main()
