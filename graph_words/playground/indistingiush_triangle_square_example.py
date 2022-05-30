from code2seq.utils import compute
import numpy
from torch.utils.data import Dataset


import networkx as nx
import torch_geometric

import visualization
from args_parse import get_default_args
from data import dataloader_utils
from graph_words import word_graphs
from graph_words.word_graphs import join_graphs, create_pyg_graph
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator
from train import training

# graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4),word_graphs.Clique(4),word_graphs.Clique(5),word_graphs.Clique(6)]
graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4)]


# graphs = [word_graphs.Cycle(3), word_graphs.Clique(4), word_graphs.Clique(5)]

class WordsCombinationGraphDataset(Dataset):
    def __init__(self, num_samples):
        self.word_graphs = [word_graphs.Cycle(3), word_graphs.Cycle(4)]
        self.name_2_label = {graph.name: i for i, graph in enumerate(self.word_graphs)}
        self.label_2_name = {i: graph.name for i, graph in enumerate(self.word_graphs)}
        self.dataset = []

        for i in range(num_samples):
            # selected_words = numpy.random.choice(word_graphs, words_per_sample).tolist()
            graph = join_graphs([word_graphs.Cycle(3), word_graphs.Cycle(4), word_graphs.Cycle(3)])
            graph2 = join_graphs([word_graphs.Cycle(3), word_graphs.Cycle(4), word_graphs.Cycle(3)])
            graph = nx.disjoint_union(graph, graph2)
            graph.add_edges_from([
                (11, 14),
                (15, 18),
                (5, 8),
                (1, 4),
                (9, 19),
                (0, 10)
            ]
            )
            pyg_graph = create_pyg_graph(graph, self.name_2_label)
            # t, p = nx.check_planarity(graph)
            # assert t
            # pyg_graph.graph.positions = nx.combinatorial_embedding_to_pos(p)
            # pyg_graph.graph.positions = {i: (i % 10, -0.5 * (i % 2) - float(i >= 10)) for i in graph}
            degrees = [y for x, y in graph.degree]
            # assert all have degree 3
            for deg in degrees:
                assert deg == 3
            self.dataset.append(pyg_graph)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


dataset = WordsCombinationGraphDataset(num_samples=1)

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


training.full_train_flow(args, device, evaluator, model, loader, loader, loader, 'coloring',
                         'acc')
