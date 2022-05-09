import global_config
import visualization
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


def visualize_activations_and_grads(model, exp):
    if not global_config.print_parameter_norms:
        return

    def calc_norm(tensor):
        return torch.norm(tensor) if tensor is not None else None

    metrics = {}
    for name, param in model.named_parameters():
        weight_norm = calc_norm(param.data)
        grad_norm = calc_norm(param.grad)
        grad_magnitude = (grad_norm / weight_norm) if (weight_norm is not None and grad_norm is not None) else None

        metrics[f'{name}:weight_norm'] = weight_norm
        metrics[f'{name}:grad_norm'] = grad_norm
        metrics[f'{name}:grad/weight'] = grad_magnitude

    if exp:
        exp.log_metrics(metrics)
    else:
        print('\n----START ACTIVATIONS------')
        print(metrics)
        print('_________________')
        print('----END ACTIVATIONS------')


def train_epoch(model, device, loader, optimizer, task_type, assert_no_zero_grad=False, grad_accum_steps=1,
                experiment=None):
    # here node_classification actually refers to sbm prediction(different loss) and coloring refers to regular node prediction,
    ## with CrossEntropy loss. Leaving it like that for now for time...
    assert task_type in {'binary classification', 'regression', 'node classification',
                         'coloring'}, f'{task_type} not supported'
    model.train()
    losses = []
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or (batch.batch[-1] == 0 and batch.num_graphs > 1):
            pass
        else:
            pred = model(batch)

            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            y = batch.y
            if y.dim() == 1:
                y = y.unsqueeze(1)
            if 'coloring' == task_type:
                loss = torch.nn.CrossEntropyLoss()(pred.squeeze(), y.squeeze())
            elif 'node classification' == task_type:
                assert is_labeled.all()
                loss = sbm_loss(pred.squeeze(), y.squeeze())
            elif "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss = loss / grad_accum_steps
            loss.backward()

            should_step = (step + 1) % grad_accum_steps == 0
            if should_step:
                visualize_activations_and_grads(model, experiment)
                if assert_no_zero_grad:
                    _assert_no_zero_grad(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
    return sum(losses) / len(losses)


def _assert_no_zero_grad(model):
    for name, p in model.named_parameters():
        if (p.grad == 0).all():
            raise Exception(f'found all zero grads in {name}, shape: {p.shape}')


class Visualizer:
    def __init__(self, task_name, epoch, graph_indexes=[0, 1]):
        self.epoch = epoch
        self.task_type = task_name
        self.graph_indexes = graph_indexes

    @staticmethod
    def _lexo(n, max_val):
        s = str(n)
        n = n * 10
        while n <= max_val:
            s = '0' + s
            n = n * 10
        return s

    def __call__(self, graphs, y_true, y_pred):
        if not self.task_type == 'coloring':
            return

        graph_to_prediction_indexes = {}
        predictions_start = 0
        for i, graph in enumerate(graphs):
            predictions_end = predictions_start + graph.num_nodes
            graph_to_prediction_indexes[i] = (predictions_start, predictions_end)
            predictions_start = predictions_end

        for index in self.graph_indexes:
            try:
                g = graphs[index]

                pred_start, pred_end = graph_to_prediction_indexes[index]
                g_pred = y_pred[pred_start:pred_end]
                g_acc = (g_pred.argmax(dim=-1) == g.y).float().mean()
                g_acc = str(round(g_acc.item(), 2))
                if self.epoch == 1:
                    visualization.draw_pyramid(g, 'x', 'input')
                    visualization.draw_pyramid(g, 'y', 'gold')
                visualization.draw_pyramid(g, g_pred, f'epoch {self._lexo(self.epoch, 1000)}. acc:{g_acc}')
            except Exception as e:
                print(f'failed visualizing {e}')


def full_train_flow(args, device, evaluator, model, test_loader, train_loader, valid_loader, task_type, eval_metric):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if args.scheduler_use_max else 'min',
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
        epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type,
                                     grad_accum_steps=args.grad_accum_steps, experiment=exp)

        print('Evaluating...')
        visualizer = Visualizer(task_name=task_type, epoch=epoch)
        valid_perf = evaluate(model, device, valid_loader, evaluator, visualizer)
        test_perf = evaluate(model, device, test_loader, evaluator)
        if global_config.log_train_acc:
            train_pref = evaluate(model, device, train_loader, evaluator)

        print(f'epoch loss {epoch_avg_loss}')
        print({'Validation': valid_perf, 'Test': test_perf})

        validation_score: float = valid_perf[eval_metric]
        test_score = test_perf[eval_metric]
        train_losses.append(epoch_avg_loss)
        valid_curve.append(validation_score)
        test_curve.append(test_score)

        exp.log_metric(f'epoch_loss', epoch_avg_loss)
        if global_config.log_train_acc:
            exp.log_metric(f'train_{eval_metric}', train_pref[eval_metric])

        exp.log_metric(f'val_{eval_metric}', validation_score)
        exp.log_metric(f'test_{eval_metric}', test_score)
        exp.log_metric('learning_rate_', optimizer.param_groups[0]['lr'])
        for key in valid_perf:
            if key.startswith('acc_'):
                exp.log_metric(key, valid_perf[key])

        # scheduler.step(validation_score - 0.00001 * epoch)

        scheduler.step(validation_score)

        if validation_score > best_so_far:
            best_so_far = validation_score
            steps_with_no_improvement = 0
        else:
            steps_with_no_improvement += 1
            if steps_with_no_improvement > args.patience:
                break
    if 'classification' in task_type or 'coloring' in task_type:
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
