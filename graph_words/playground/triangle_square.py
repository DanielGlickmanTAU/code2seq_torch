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

graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4)]
dataset = word_graphs.WordsCombinationGraphDataset(graphs, num_samples=1,
                                                   words_per_sample=2)[0]

# overfit train
args = get_default_args()

torch_geometric.seed_everything(args.seed)

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

args.normalize = False
stack = AdjStack(args)
dataset = stack(dataset)
adj_matrix_normalized_rows_of_2_nodes_between_circles = dataset.adj_stack[1][2:4]
print(dataset)
