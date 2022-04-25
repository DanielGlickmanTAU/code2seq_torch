import networkx as nx
import matplotlib.pyplot as plt
import tabulate

from code2seq.utils import compute

torch = compute.get_torch()
from coloring.coloring_utils import tensor_to_tuple
from coloring.datasets import PyramidEdgeColorDataset
from tests.test_utils import MockModule

from torch.utils.data import DataLoader
import torch_geometric


def acc_at_init(loader):
    model.eval()
    for step, (x, y, _) in enumerate(loader):
        y_hat = (model(x).sigmoid() > 0.5).squeeze()
        correct = y_hat == y
        return sum(correct) / len(correct)


def train_eval_loop(train_loader, eval_loader, model, draw_every=100, stop_thershold=1.):
    acc_at_epoch_500, fp_at_epoch_500 = 0., 0.
    acc_at_epoch_2k, fp_at_epoch_2k = 0., 0.
    best_acc, best_fp, = 0., 0.

    losses = []
    correct_all = []
    train_correct_all = []
    false_positive_all = []
    val_predictions = []

    for step, (x, y, graph_nodes) in enumerate(train_loader):
        y_hat = model(x)
        loss = torch.nn.BCEWithLogitsLoss()(y_hat, y.float().reshape(y_hat.shape))
        losses.append(loss.item())
        correct = (y_hat.sigmoid() > 0.5).squeeze() == y
        train_correct_all.extend(correct.tolist())

    for step, (x, y, graph_nodes) in enumerate(eval_loader):
        graph_nodes = PyramidEdgeColorDataset.tensor_to_node_indexes(graph_nodes)
        logits = model(x, print_p=True)
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
        if acc > best_acc:
            best_acc = acc
        acc_train = sum(train_correct_all) / len(train_correct_all)
        fp = sum(false_positive_all) / len(false_positive_all)

    # sort by highest loss
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

    return {'final_acc': acc, 'final_acc_train': acc_train, 'final_fp': fp,

            'best_acc': best_acc, 'best_fp': best_fp}


torch_geometric.seed_everything(1)


def get_model(num_adj_stacks, hidden_layer_multiplier, use_batch_norm, just_sum, network_then_projection,
              normalize=None):
    hidden_dim = hidden_layer_multiplier * num_adj_stacks

    first_layer = torch.nn.Sequential(
        torch.nn.Linear(num_adj_stacks, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, num_adj_stacks),
    ) if network_then_projection else torch.nn.Identity()

    model = torch.nn.Sequential(
        first_layer,
        MockModule(lambda x: torch.nn.functional.normalize(
            x)) if normalize == 'l2' else torch.nn.LayerNorm(
            num_adj_stacks) if normalize == 'layer' else torch.nn.Identity(),
        torch.nn.Linear(num_adj_stacks, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim) if use_batch_norm else torch.nn.Identity(),
        torch.nn.ReLU(),
        MockModule(lambda x: x.sum(dim=-1)) if just_sum else torch.nn.Linear(hidden_dim, 1),
    )
    return model


table = [
    ['pyramid base train', 'pyramid base test', 'edge size',
     'final acc', 'final FP', 'final_acc_train', 'best acc', 'best fp']

]
device = compute.get_device()


class Oracle(torch.nn.Module):
    def __init__(self, dataset):
        super(Oracle, self).__init__()
        self.ground_truth = {tensor_to_tuple(edge): float(label.item()) for edge, label, _ in dataset.dataset}

    def forward(self, x, print_p=False):
        x = [tensor_to_tuple(sample) for sample in x]
        if print_p:
            print(
                f'amount of edges in test that are in train: {sum([1 for i in x if i in self.ground_truth]) / len(x)}')
        return torch.tensor([self.ground_truth[sample] if sample in self.ground_truth else 0. for sample in x])


for max_row_size in [10]:
    # for max_row_size in [4]:
    # for edge_size_plus in [0, 1, 2]:
    # for edge_size_plus in [-3, -2, -1, 0, 1]:
    for test_size_plus in [0, 1, 2, 3, 4]:
        for edge_size_plus in [-6, -5, -4, -3, -2, -1, 0]:
            num_adj_stacks = max_row_size + edge_size_plus
            dataset = PyramidEdgeColorDataset(max_row_size, num_adj_stacks)
            loader = DataLoader(dataset, batch_size=64000, shuffle=True)
            model = Oracle(dataset)

            test_pyramid_base = max_row_size + test_size_plus
            # test_pyramid_base = max_row_size
            dataset = PyramidEdgeColorDataset(test_pyramid_base, num_adj_stacks)
            test_loader = DataLoader(dataset, batch_size=64000, shuffle=True)

            num_params = sum(p.numel() for p in model.parameters())

            d = train_eval_loop(loader, test_loader, model, draw_every=1000)
            table.append(
                [max_row_size, test_pyramid_base, num_adj_stacks,

                 d['final_acc'], d['final_fp'], d['final_acc_train'], d['best_acc'], d['best_fp'],

                 ]
            )
            print(tabulate.tabulate(table))
            print('\n\n\n')

print(tabulate.tabulate(table))
print('')
