from args_parse import add_args
from code2seq.utils import compute
from dataset_transformations import DistanceCalculator, AdjStack

from exp_utils import start_exp
from torchvision import transforms

torch = compute.get_torch()
from pytorch_lightning.loggers import CometLogger
from torch_geometric.loader import DataLoader
import torch.optim as optim

from gnn import GNN
from tqdm import tqdm
import argparse
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train_epoch(model, device, loader, optimizer, task_type):
    model.train()
    losses = []
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
            losses.append(loss.item())
    return sum(losses) / len(losses)


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
    add_args(parser)
    AdjStack.add_args(parser)
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      transform=transforms.Compose([DistanceCalculator(), AdjStack(args)]))

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

    model = get_model(args, dataset, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    valid_curve = []
    test_curve = []
    train_losses = []
    best_so_far = 0.
    steps_with_no_improvement = 0

    exp = start_exp(args.exp_name, args, model)
    try:
        print(f'#Params: {sum(p.numel() for p in model.parameters())}')
    except:
        print('fail print params')

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        # train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print(f'epoch loss {epoch_avg_loss}')
        print({'Validation': valid_perf, 'Test': test_perf})

        validation_score = valid_perf[dataset.eval_metric]
        test_score = test_perf[dataset.eval_metric]
        train_losses.append(epoch_avg_loss)
        valid_curve.append(validation_score)
        test_curve.append(test_score)

        # train_score = train_perf[dataset.eval_metric]
        exp.log_metric(f'epoch_loss', epoch_avg_loss)
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
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
    best_train = min(train_losses)

    train_perf = eval(model, device, train_loader, evaluator)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    exp.log_metric(f'last_test_', test_curve[best_val_epoch])
    train_score = train_perf[dataset.eval_metric]
    exp.log_metric(f'train_{dataset.eval_metric}', train_score)

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_losses[best_val_epoch], 'BestTrain': best_train}, args.filename)


def get_model(args, dataset, device):
    if args.gnn == 'gin':
        model = GNN(args, gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False, num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(args, gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True, num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gcn':
        model = GNN(args, gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False, num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(args, gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True, num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual).to(device)
    else:
        raise ValueError('Invalid GNN type')
    return model


if __name__ == "__main__":
    main()
