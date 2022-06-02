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


def add_args(parser):
    parser.add_argument('--coloring_mode', type=str,
                        help='coloring mode for task. either global or instance')
    parser.add_argument('--atoms_set', type=int, help='one of predifined sets of atoms we test on')


args = get_default_args(add_args)
coloring_mode = args.coloring_mode
num_colors = args.num_colors
assert coloring_mode == 'global' or coloring_mode == 'instance', f'got {coloring_mode}'
atom_set = args.atoms_set

graphs = word_graphs.get_atom_set(atom_set)

dataset = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=2000,
                                                   words_per_sample=4, num_rows=4, num_colors=num_colors)

dataset_val = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=250,
                                                       words_per_sample=4, num_rows=4, num_colors=num_colors)
dataset_train = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=250,
                                                         words_per_sample=4, num_rows=4, num_colors=num_colors)

torch_geometric.seed_everything(args.seed)

args.num_layer = 6

args.num_transformer_layers = 0
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.num_heads = 1
args.patience = 100
args.epochs = 2000
# args.lr_schedule_patience = 500
args.lr_reduce_factor = 0.9
args.conv_track_running_stats = False
device = compute.get_device()
task = 'coloring'
num_labels = dataset.num_labels
model = model_utils.get_model(args, num_tasks=num_labels, device=device, task=task, num_embedding=num_labels + 1)
evaluator = Evaluator('coloring')
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args))
valid_loader = dataloader_utils.create_dataset_loader(dataset_val, batch_size=64, mapping=AdjStack(args), shuffle=True)
test_loader = dataloader_utils.create_dataset_loader(dataset_train, batch_size=64, mapping=AdjStack(args),
                                                     shuffle=True)

training.full_train_flow(args, device, evaluator, model, test_loader, loader, valid_loader, 'coloring',
                         'acc')
