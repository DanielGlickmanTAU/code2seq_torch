from code2seq.utils import compute
from unittest import TestCase


import visualization

from args_parse import get_default_args
from data import dataloader_utils
from graph_words import word_graphs
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator
from tests import test_flow_utils
from train import training

class Test(TestCase):
    def test_join_graphs(self):
        clique5 = word_graphs.clique_5
        cycle4 = word_graphs.cycle_4
        g = word_graphs.join_graphs([clique5, cycle4])
        visualization.draw_pyg_graph(g)
        self.assertEqual(len(g), len(cycle4) + len(clique5))

    def test_single_word_graphs_overfit_4cycle_vs_clique_with_gnn(self):


        dataset = word_graphs.WordGraphDataset([word_graphs.Cycle(4),word_graphs.Clique(4)])

        args = get_default_args()

        args.num_transformer_layers = 0
        args.num_layer = 4
        args.drop_ratio = 0.
        args.transformer_encoder_dropout = 0.
        args.emb_dim = 30
        # args.patience = 4000
        # args.lr_schedule_patience = 40
        args.epochs = 300
        # args.lr_schedule_patience = 500
        args.lr_reduce_factor = 0.9

        args.conv_track_running_stats = False

        num_colors = len(dataset.name_2_label)
        device = compute.get_device()
        task = 'coloring'
        model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task,
                                      num_embedding=num_colors + 1)
        evaluator = Evaluator('coloring')
        valid_loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args),
                                                              shuffle=False)

        test_flow_utils.train_and_assert_overfit(model, valid_loader, evaluator, 'coloring')

        # training.full_train_flow(args, device, evaluator, model, valid_loader, valid_loader, valid_loader, 'coloring',
        #                          'acc')
