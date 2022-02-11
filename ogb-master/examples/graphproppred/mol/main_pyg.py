from code2seq.utils import compute  ##import this first
import torch_geometric

from args_parse import get_default_args
from data.dataloader_utils import get_train_val_test_loaders
from dataset_transformations import DistanceCalculator
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack

from exp_utils import start_exp
from torchvision import transforms

from train import train_epoch, evaluate

torch = compute.get_torch()
import torch.optim as optim

import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def main():
    # Training settings
    args = get_default_args()
    args.dataset = "ogbg-molhiv"

    torch_geometric.seed_everything(args.seed)

    device = compute.get_device()

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      transform=transforms.Compose([DistanceCalculator(), AdjStack(args)]))

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    test_loader, train_loader, valid_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                         batch_size=args.batch_size)

    model = get_model(args, dataset.num_tasks, device, task='mol')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    valid_curve = []
    test_curve = []
    train_losses = []
    best_so_far = 0.
    steps_with_no_improvement = 0

    exp = start_exp(args.exp_name, args, model)
    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        # train_perf = evaluate(model, device, train_loader, evaluator)
        valid_perf = evaluate(model, device, valid_loader, evaluator)
        test_perf = evaluate(model, device, test_loader, evaluator)

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

    train_perf = evaluate(model, device, train_loader, evaluator)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    exp.log_metric(f'last_test_', test_curve[best_val_epoch])
    train_score = train_perf[dataset.eval_metric]
    exp.log_metric(f'train_{dataset.eval_metric}', train_score)


if __name__ == "__main__":
    main()
