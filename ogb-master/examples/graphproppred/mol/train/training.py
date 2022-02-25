from exp_utils import start_exp
import numpy as np
from torch import optim as optim
from tqdm import tqdm
import torch

# from train import evaluate
from train.eval import evaluate
from train.loss import sbm_loss

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train_epoch(model, device, loader, optimizer, task_type, assert_no_zero_grad=False):
    assert task_type in {'binary classification', 'regression', 'node classification'}, f'{task_type} not supported'
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
            y = batch.y
            if y.dim() == 1:
                y = y.unsqueeze(1)
            if 'node classification' == task_type:
                assert is_labeled.all()
                loss = sbm_loss(pred.squeeze(), y.squeeze())
            elif "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            if assert_no_zero_grad:
                _assert_no_zero_grad(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
    return sum(losses) / len(losses)


def _assert_no_zero_grad(model):
    for name, p in model.named_parameters():
        if (p.grad == 0).all():
            raise Exception(f'found all zero grads in {name}, shape: {p.shape}')


def full_train_flow(args, device, evaluator, model, test_loader, train_loader, valid_loader, task_type, eval_metric):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_reduce_factor,
                                                     patience=args.lr_schedule_patience,
                                                     verbose=True)
    exp = start_exp(args.exp_name, args, model)

    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    valid_curve = []
    test_curve = []
    train_losses = []
    best_so_far = 0.
    steps_with_no_improvement = 0

    print_first_valid_loss(device, evaluator, model, valid_loader)
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type)

        print('Evaluating...')
        valid_perf = evaluate(model, device, valid_loader, evaluator)
        test_perf = evaluate(model, device, test_loader, evaluator)

        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print(f'epoch loss {epoch_avg_loss}')
        print({'Validation': valid_perf, 'Test': test_perf})

        validation_score: float = valid_perf[eval_metric]
        test_score = test_perf[eval_metric]
        train_losses.append(epoch_avg_loss)
        valid_curve.append(validation_score)
        test_curve.append(test_score)

        exp.log_metric(f'epoch_loss', epoch_avg_loss)
        exp.log_metric(f'val_{eval_metric}', validation_score)
        exp.log_metric(f'test_{eval_metric}', test_score)
        exp.log_metric('learning_rate', optimizer.param_groups[0]['lr'])

        scheduler.step(validation_score - 0.00001 * epoch)

        if validation_score > best_so_far:
            best_so_far = validation_score
            steps_with_no_improvement = 0
        else:
            steps_with_no_improvement += 1
            if steps_with_no_improvement > args.patience:
                break
    if 'classification' in task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
    train_perf = evaluate(model, device, train_loader, evaluator)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    exp.log_metric(f'last_test_', test_curve[best_val_epoch])
    train_score = train_perf[eval_metric]
    exp.log_metric(f'train_{eval_metric}', train_score)


def print_first_valid_loss(device, evaluator, model, valid_loader):
    valid_perf = evaluate(model, device, valid_loader, evaluator)
    print(f'first valid score is {valid_perf}')
