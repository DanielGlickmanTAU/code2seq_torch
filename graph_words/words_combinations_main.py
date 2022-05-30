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

# graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4),word_graphs.Clique(4),word_graphs.Clique(5),word_graphs.Clique(6)]
graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4)]
# graphs = [word_graphs.Cycle(3), word_graphs.Clique(4), word_graphs.Clique(5)]
dataset = word_graphs.WordsCombinationGraphDataset(graphs, num_samples=1000,
                                                   words_per_sample=10)

dataset_val = word_graphs.WordsCombinationGraphDataset(graphs, num_samples=100,
                                                       words_per_sample=8)

dataset_train = word_graphs.WordsCombinationGraphDataset(graphs, num_samples=100,
                                                         words_per_sample=12)

# overfit train
args = get_default_args()

torch_geometric.seed_everything(args.seed)

# num_adj_stacks = pyramid_size - 1

args.num_transformer_layers = 0
args.num_layer = 6
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.num_heads = 1
args.patience = 400
args.epochs = 2000
# args.lr_schedule_patience = 500
args.lr_reduce_factor = 0.9
args.conv_track_running_stats = False
args.gnn = 'gcn'
num_colors = len(dataset.name_2_label)
device = compute.get_device()
task = 'coloring'
model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task, num_embedding=num_colors + 1)
evaluator = Evaluator('coloring')
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args))
valid_loader = dataloader_utils.create_dataset_loader(dataset_val, batch_size=64, mapping=AdjStack(args), shuffle=True)
test_loader = dataloader_utils.create_dataset_loader(dataset_train, batch_size=64, mapping=AdjStack(args),
                                                     shuffle=True)

training.full_train_flow(args, device, evaluator, model, test_loader, loader, valid_loader, 'coloring',
                         'acc')
