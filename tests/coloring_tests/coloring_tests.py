import unittest
from unittest import TestCase

from code2seq.utils import compute
import exp_utils
from torch import optim
from torch_geometric.loader import DataLoader

from args_parse import get_default_args
from coloring.datasets import PyramidNodeColorDataset
import visualization
from coloring import coloring_utils
from data import dataloader_utils
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator
from tests import test_flow_utils

num_colors = 3
device = compute.get_device()
task = 'coloring'
batch_size = 256
batch_size_test = 2 * batch_size


class Test(TestCase):
    def test_gnn_train_5_test_6(self):
        pyramid_size = 5
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 0
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')
        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = None
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)

    def test_transformer_train_5_test_6(self):
        pyramid_size = 5
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 4
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')
        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = Noneda
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)

    def test_position_transformer_train_5_test_6(self):
        pyramid_size = 5
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 4
        args.attention_type = 'position'
        args.use_ffn_for_attention_weights = True
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.num_heads = 1
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')
        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = Noneda
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)


    def test_position_transformer_train_10_test_11(self):
        pyramid_size = 5
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 4
        args.attention_type = 'position'
        args.use_ffn_for_attention_weights = True
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.num_heads = 1
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')
        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = Noneda
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)


    def test_gnn_train_10_test_11(self):
        pyramid_size = 10
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 0
        args.num_layer = 6
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')

        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = None
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)

    def test_gnn_train_15_test_16(self):
        pyramid_size = 15
        num_adj_stacks = pyramid_size + 1
        dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
        dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

        args = get_default_args()
        args.num_transformer_layers = 0
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 100
        args.adj_stack = list(range(num_adj_stacks))

        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')

        loader = dataloader_utils.create_dataset_loader(dataset, batch_size=batch_size, mapping=AdjStack(args))
        test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=batch_size_test,
                                                             mapping=AdjStack(args),
                                                             shuffle=False)
        # exp = None
        exp = exp_utils.start_exp("test", args, model)

        test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)


if __name__ == '__main__':
    unittest.main()
