from arg_parse_utils import bool_
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
    parser.add_argument('--atoms_set', type=int, help='one of pre difined sets of atoms we test on')
    parser.add_argument('--num_colors', type=int)
    parser.add_argument('--edge_p', type=float, default=1.)
    parser.add_argument('--only_color', type=bool_, default=False)
    parser.add_argument('--unique_atoms_per_example', type=bool_, default=False)


args = get_default_args(add_args)
coloring_mode = args.coloring_mode
num_colors = args.num_colors
atom_set = args.atoms_set
edge_p = args.edge_p
unique_atoms_per_example = args.unique_atoms_per_example

row_size = 4
# args.num_layer = 16
only_color = args.only_color

graphs = word_graphs.get_atom_set(atom_set)

n_train = 3000
n_valid = 1000
# n_train = 20
# n_valid = 10
dataset = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=n_train,
                                                   words_per_sample=row_size, num_rows=row_size, num_colors=num_colors,
                                                   only_color=only_color,unique_atoms_per_example= unique_atoms_per_example)

dataset_val = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=n_valid,
                                                       words_per_sample=row_size, num_rows=row_size,
                                                       num_colors=num_colors, only_color=only_color,unique_atoms_per_example= unique_atoms_per_example)
dataset_train = word_graphs.WordsCombinationGraphDataset(coloring_mode, graphs, num_samples=300,
                                                         words_per_sample=row_size, num_rows=row_size,
                                                         num_colors=num_colors, only_color=only_color,unique_atoms_per_example= unique_atoms_per_example)

torch_geometric.seed_everything(args.seed)

args.num_transformer_layers = 0
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.num_heads = 1
args.patience = 250
args.epochs = 2000
# args.lr_schedule_patience = 500
args.lr_reduce_factor = 0.9
args.conv_track_running_stats = False
device = compute.get_device()
# task = 'coloring'
task = 'PATTERN'
# task_type = 'coloring'
task_type = 'node classification'
num_labels = dataset.num_labels
assert coloring_mode == 'rows', 'else need to restore task_type and task.'
model = model_utils.get_model(args, num_tasks=num_labels, device=device, task='coloring', num_embedding=num_colors + 1)
evaluator = Evaluator(task)
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args), shuffle=True)
valid_loader = dataloader_utils.create_dataset_loader(dataset_val, batch_size=64, mapping=AdjStack(args), shuffle=False)
test_loader = dataloader_utils.create_dataset_loader(dataset_train, batch_size=64, mapping=AdjStack(args),
                                                     shuffle=False)

training.full_train_flow(args, device, evaluator, model, test_loader, loader, valid_loader, task_type,
                         'acc')
print(f'task  {task_type}')
