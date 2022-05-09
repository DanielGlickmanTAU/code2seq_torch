from code2seq.utils import compute

import networkx as nx
import torch_geometric

import visualization
from args_parse import get_default_args
from data import dataloader_utils
from graph_words import word_graphs
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator
from train import training

dataset = word_graphs.WordGraphDataset()

# # draw graphs, each should have different color(by label)
# for pyg_graph in dataset:
#     visualization.draw(pyg_graph, pyg_graph.y, color_map={0: 'pink', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow'})
# # color with x(no label) just as sanity
# visualization.draw(pyg_graph, pyg_graph.x, color_map={0: 'gray'})

# overfit train
args = get_default_args()

torch_geometric.seed_everything(args.seed)

# num_adj_stacks = pyramid_size - 1

args.num_transformer_layers = 0
args.num_layer = 5
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.num_heads = 1
args.patience = 100
args.epochs = 2000
# args.lr_schedule_patience = 500
args.lr_reduce_factor = 0.9
num_colors = len(dataset.name_2_label)
device = compute.get_device()
task = 'coloring'
model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task, num_embedding=num_colors + 1)
evaluator = Evaluator('coloring')
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args))
valid_loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args), shuffle=False)
test_loader = dataloader_utils.create_dataset_loader(dataset, batch_size=32, mapping=AdjStack(args),
                                                     shuffle=False)
# exp = None
# exp = exp_utils.start_exp(args.exp_name, args, model)
# test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)

training.full_train_flow(args, device, evaluator, model, test_loader, loader, valid_loader, 'coloring',
                         'acc')

# join graphs

# join edges(last node to first node...)
