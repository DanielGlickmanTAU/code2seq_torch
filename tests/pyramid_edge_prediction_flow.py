import networkx as nx
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import Dataset

import coloring.graph_generation
from code2seq.utils import compute
from code2seq.utils.StoppingCritertion import StoppingCriterion
from coloring.coloring_utils import color_graph, create_stacks, \
    map_tensor_edge_to_color, map_tensor_edge_to_networkx_node_ids

torch = compute.get_torch()

from torch.utils.data import DataLoader
import torch_geometric


class PyramidEdgeColorDataset(Dataset):
    """ creates a dataset where the inputs are random walk probabilities edges of a single pyramid graph
    and the labels are True/False if the edge connects between nodes of the same color"""

    def __init__(self, max_row_size, num_adj_stack):
        graph, positions = coloring.graph_generation.create_pyramid(1, max_row_size)
        color_graph(graph)
        self.graph = graph
        self.positions = positions

        data = torch_geometric.utils.from_networkx(graph)
        stacks = create_stacks(data, num_adj_stack)
        stacks = stacks.permute(1, 2, 0)
        edge_to_node_ids = map_tensor_edge_to_networkx_node_ids(graph, stacks)

        self.dataset = []
        for edge_tensor, list_of_node_ids in edge_to_node_ids.items():
            edge_tensor = torch.tensor(edge_tensor)
            for node_i, node_j in list_of_node_ids:
                same_color = graph.nodes[node_i]['color'] == graph.nodes[node_j]['color']
                self.dataset.append(
                    (edge_tensor, torch.tensor(same_color), self.node_index_tuples_to_tensor(node_i, node_j)))

    @staticmethod
    # node_i,node_j are also tuples of location/index e.g (1,0)
    def node_index_tuples_to_tensor(node_i, node_j):
        return torch.tensor((node_i, node_j))

    @staticmethod
    def tensor_to_node_indexes(tensor):
        if tensor.dim() == 3:
            return [PyramidEdgeColorDataset.tensor_to_node_indexes(e) for e in tensor]

        assert tensor.shape == (2, 2)
        return tuple(tensor[0].numpy()), tuple(tensor[-1].numpy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def acc_at_init(loader):
    model.eval()
    for step, (x, y, _) in enumerate(loader):
        y_hat = (model(x).sigmoid() > 0.5).squeeze()
        correct = y_hat == y
        return sum(correct) / len(correct)


torch_geometric.seed_everything(1)
max_row_size = 4
num_adj_stacks = max_row_size + 1
dataset = PyramidEdgeColorDataset(max_row_size, num_adj_stacks)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

hidden_dim = 4 * num_adj_stacks
model = torch.nn.Sequential(

    torch.nn.Linear(num_adj_stacks, hidden_dim),
    torch.nn.BatchNorm1d(hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, num_adj_stacks),
    torch.nn.Linear(num_adj_stacks, 1),
)
# model = torch.nn.Sequential(
#     torch.nn.Linear(num_adj_stacks, 1)
# )
# model[-1].bias.data = torch.tensor([-0.1])
# model = torch.nn.Linear(num_adj_stacks,1)

epoch = 10000
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = StoppingCriterion(threshold=1., desired_stable_evaluations=10)
last_acc = -1
print(f'acc at init {acc_at_init(loader)}')
for i in range(epoch):

    losses = []
    correct_all = []
    false_positive_all = []
    val_predictions = []

    model.train()
    for step, (x, y, graph_nodes) in enumerate(loader):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = torch.nn.BCEWithLogitsLoss()(y_hat, y.float().reshape(y_hat.shape))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    for step, (x, y, graph_nodes) in enumerate(loader):
        graph_nodes = PyramidEdgeColorDataset.tensor_to_node_indexes(graph_nodes)
        logits = model(x)
        y_hat = (logits.sigmoid() > 0.5).squeeze()
        correct = y_hat == y
        correct_all.extend(correct.tolist())
        false_positive = y_hat.logical_and(~y)
        false_positive_all.extend(false_positive.tolist())

        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y.float(),
                                                                        reduction='none').squeeze()
        for loss, sample, nodes, is_correct, logit, label in zip(val_loss, x, graph_nodes, correct, logits, y):
            val_predictions.append(
                {'loss': loss, 'sample': sample, 'nodes': nodes, 'is_correct': is_correct, 'logit': logit,
                 'label': label})
        # val_predictions.extend()
    acc = sum(correct_all) / len(correct_all)
    fp = sum(false_positive_all) / len(false_positive_all)

    # sort by highest loss
    if acc != last_acc and i % 100 == 0:
        last_acc = acc
        sorted_val_predictions = sorted(val_predictions, key=lambda d: -d['loss'])
        wrong_predictions = [x for x in val_predictions if not x['is_correct']]
        wrong_edges = {d['nodes'] for d in wrong_predictions}
        mg = nx.MultiDiGraph(dataset.graph)
        mg.add_edges_from(wrong_edges)
        edge_colors = ['red' if edge in wrong_edges else 'black' for edge in mg.edges()]
        nx.draw(mg, dataset.positions,
                node_color=[dataset.graph.nodes[x]['color'] for x in dataset.graph.nodes()],
                edge_color=edge_colors,
                with_labels=True)
        plt.title(f'acc:{acc}')
        plt.show()

    print(f' epoch {i}, loss {sum(losses) / len(losses)}, acc:{acc}, false positive {fp}')
    if criterion(acc):
        break

print('a')
