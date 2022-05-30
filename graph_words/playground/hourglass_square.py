import networkx as nx
import numpy
import torch_geometric

from args_parse import get_default_args
from graph_words import word_graphs
from model.positional.positional_attention_weight import AdjStack
import visualization

graph = nx.disjoint_union(word_graphs.HourGlass(), word_graphs.JoinedSquared())
graph = torch_geometric.utils.from_networkx(graph)
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
args.adj_stacks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# args.normalize = False
stack = AdjStack(args)
dataset = stack(graph)
ones = numpy.ones((dataset.num_nodes,1))


x = dataset.adj_stack[2].T @ ones
print(x)