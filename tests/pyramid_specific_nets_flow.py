import networkx as nx
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import Dataset
import tabulate

import coloring.graph_generation
from code2seq.utils import compute
from code2seq.utils.StoppingCritertion import StoppingCriterion
from coloring.color_datasets import PyramidEdgeColorDataset
from coloring.coloring_utils import color_graph, create_stacks, \
    map_tensor_edge_to_color, map_tensor_edge_to_networkx_node_ids
from tests.test_utils import MockModule

torch = compute.get_torch()

from torch.utils.data import DataLoader
import torch_geometric


def acc_at_init(loader):
    model.eval()
    for step, (x, y, _) in enumerate(loader):
        y_hat = (model(x).sigmoid() > 0.5).squeeze()
        correct = y_hat == y
        return sum(correct) / len(correct)


def train_eval_loop(train_loader, eval_loader, model, draw_every=100, stop_thershold=1.):
    epoches = 100000
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = StoppingCriterion(threshold=stop_thershold, desired_stable_evaluations=10, patience=2_000)
    last_acc = -1
    acc_at_epoch_500, fp_at_epoch_500 = 0., 0.
    acc_at_epoch_2k, fp_at_epoch_2k = 0., 0.
    best_acc, best_fp, = 0., 0.
    for epoch in range(epoches):

        losses = []
        correct_all = []
        train_correct_all = []
        false_positive_all = []
        val_predictions = []

        model.train()
        for step, (x, y, graph_nodes) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = torch.nn.BCEWithLogitsLoss()(y_hat, y.float().reshape(y_hat.shape))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct = (y_hat.sigmoid() > 0.5).squeeze() == y
            train_correct_all.extend(correct.tolist())

        model.eval()
        for step, (x, y, graph_nodes) in enumerate(eval_loader):
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
        acc_train = sum(train_correct_all) / len(train_correct_all)
        fp = sum(false_positive_all) / len(false_positive_all)

        if epoch == 500:
            acc_at_epoch_500, fp_at_epoch_500 = acc, fp
        if epoch == 2000:
            acc_at_epoch_2k, fp_at_epoch_2k = acc, fp
        if acc > best_acc:
            best_acc, best_fp = acc, fp

        if epoch == epoches - 1 or (acc != last_acc and (epoch + 1) % draw_every == 0):
            last_acc = acc
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
        if epoch % 200 == 0:
            print(
                f' epoch {epoch}, loss {sum(losses) / len(losses)},acc_train:{acc_train} acc_val:{acc}, false positive {fp}')
        if criterion(acc):
            break

    return {'final_acc': acc, 'final_acc_train': acc_train, 'final_fp': fp,
            'acc_at_epoch_500': acc_at_epoch_500, 'fp_at_epoch_500': fp_at_epoch_500,
            'acc_at_epoch_2k': acc_at_epoch_2k, 'fp_at_epoch_2k': fp_at_epoch_2k,
            'epoch_to_converge_at_1': epoch,
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
    ['pyramid base train', 'pyramid base test', 'edge size', 'hidden layer multipler', 'model_name', '#params',
     'final acc', 'final FP', 'final_acc_train', 'best acc', 'best fp', 'epochs to converge',
     'acc at epoch 500'
        , 'acc at 2k']

]
device = compute.get_device()
for max_row_size in [10]:
    # for max_row_size in [4]:
    # for edge_size_plus in [0, 1, 2]:
    for test_size_plus in [1, 2, 3, 4]:
        for edge_size_plus in [-4, -3, -2, -1, 0]:
            # for edge_size_plus in [0]:

            num_adj_stacks = max_row_size + edge_size_plus
            dataset = PyramidEdgeColorDataset(max_row_size, num_adj_stacks)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            test_pyramid_base = max_row_size + test_size_plus
            # test_pyramid_base = max_row_size
            dataset = PyramidEdgeColorDataset(test_pyramid_base, num_adj_stacks)
            test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

            for hidden_layer_multiplier in [2]:
                # vanila_model = get_model(num_adj_stacks, hidden_layer_multiplier=hidden_layer_multiplier,
                #                          use_batch_norm=True, just_sum=False, normalize=None,
                #                          network_then_projection=False)
                # layer_norm_just_sum = get_model(num_adj_stacks,
                #                                 hidden_layer_multiplier=hidden_layer_multiplier,
                #                                 use_batch_norm=True, just_sum=True,
                #                                 normalize='layer',
                #                                 network_then_projection=False)
                ffn_before_bn_ff = get_model(num_adj_stacks, hidden_layer_multiplier=hidden_layer_multiplier,
                                             use_batch_norm=True, just_sum=False, normalize=None,
                                             network_then_projection=True)
                # ffn_before_layer_norm_ff_no_bn = get_model(num_adj_stacks,
                #                                            hidden_layer_multiplier=hidden_layer_multiplier,
                #                                            use_batch_norm=False, just_sum=False,
                #                                            normalize='layer',
                #                                            network_then_projection=True)
                #
                # ffn_before_layer_norm_ff_yes_bn = get_model(num_adj_stacks,
                #                                             hidden_layer_multiplier=hidden_layer_multiplier,
                #                                             use_batch_norm=True, just_sum=False,
                #                                             normalize='layer',
                #                                             network_then_projection=True)

                models = [
                    # ('vanila_model', vanila_model),
                    # ('layer_norm_just_sum', layer_norm_just_sum),
                    ('ffn_before_bn_ff', ffn_before_bn_ff),
                    # ('ffn_before_layer_norm_ff_no_bn', ffn_before_layer_norm_ff_no_bn),
                    # ('ffn_before_layer_norm_ff_yes_bn', ffn_before_layer_norm_ff_yes_bn)
                ]

                for model_name, model in models:
                    model = model.to(device)
                    num_params = sum(p.numel() for p in model.parameters())
                    d = train_eval_loop(loader, test_loader, model, draw_every=1000)
                    table.append(
                        [max_row_size, test_pyramid_base, num_adj_stacks, hidden_layer_multiplier,
                         model_name,
                         num_params,
                         d['final_acc'], d['final_fp'], d['final_acc_train'], d['best_acc'], d['best_fp'],
                         d['epoch_to_converge_at_1'],
                         d['acc_at_epoch_500']
                            , d['acc_at_epoch_2k']
                         ]
                    )
                    print(tabulate.tabulate(table))
                    print('\n\n\n')

print(tabulate.tabulate(table))
print('')
