from code2seq.utils import compute
import comet_ml

from commode_utils.callbacks import ModelCheckpointWithUploadCallback

torch = compute.get_torch()
from pytorch_lightning.loggers import CometLogger, WandbLogger
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import graph_algos
from gnn import GNN
from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

import torch_geometric.transforms
from torch_geometric.data import Data


class DistanceCalculator(torch_geometric.transforms.BaseTransform):
    def __call__(self, data: Data):
        edge_index = data.edge_index
        N = data.x.size(0)
        # (row, col) = data.edge_index
        adj = torch.full([N, N], torch.inf)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        adj.fill_diagonal_(0)
        shortest_path = graph_algos.floyd_warshall(adj)
        # pytorch-geometric collate expects tensor with all same dims, except for the 0 dim. so we pack here to a single dim, and unpack it back in GNN#forward
        # data.distances = shortest_path.view(-1)
        data.distances = shortest_path

        return data


def train_epoch(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=6,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--num_transformer_layers', type=int, default=0,
                        help='number of transformer layers after GNN')
    parser.add_argument('--transformer_ff_dim', type=int, default=600,
                        help='transformer feedforward dim')
    parser.add_argument('--graph_pooling', type=str, default='mean',
                        help='graph pooling')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=8,
                        help='training early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4 * torch.cuda.device_count(),
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--distance_bias', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    comet_logger = CometLogger(
        project_name='graph-filter-network',
        api_key='FvAd5fm5rJLIj6TtmfGHUJm4b',
        workspace="danielglickmantau"
    )

    # need this for starting the experiment
    exp = comet_logger.experiment

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=args.dataset, transform=DistanceCalculator())
    # dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    num_transformer_layers = args.num_transformer_layers
    layer = args.num_layer - num_transformer_layers
    assert layer >= 0
    assert num_transformer_layers >= 0
    if args.gnn == 'gin':
        model = GNN(args, gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False, num_transformer_layers=num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(args, gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True, num_transformer_layers=num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gcn':
        model = GNN(args, gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False, num_transformer_layers=num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(args, gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True, num_transformer_layers=num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    valid_curve = []
    test_curve = []
    train_curve = []
    best_so_far = 0.
    steps_with_no_improvement = 0

    exp.set_model_graph(model)
    exp.log_parameters(args)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_epoch(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_score = train_perf[dataset.eval_metric]
        validation_score = valid_perf[dataset.eval_metric]
        test_score = test_perf[dataset.eval_metric]
        train_curve.append(train_score)
        valid_curve.append(validation_score)
        test_curve.append(test_score)

        exp.log_metric(f'train_{dataset.eval_metric}', train_score)
        exp.log_metric(f'val_{dataset.eval_metric}', validation_score)
        exp.log_metric(f'test_{dataset.eval_metric}', test_score)

        if validation_score > best_so_far:
            best_so_far = validation_score
            steps_with_no_improvement = 0
        else:
            steps_with_no_improvement += 1
            if steps_with_no_improvement > args.patience:
                break

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
