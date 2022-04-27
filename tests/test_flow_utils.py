from code2seq.utils import compute
from torch import optim

from train.eval import evaluate
from train.training import train_epoch


def train_and_assert_overfit(model, train_loader, evaluator, task_type, score_needed=0.9, exp=None,
                             lr=3e-5, test_loader=None):
    if test_loader is None:
        test_loader = train_loader
    device = compute.get_device()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 500 + 1):
        epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type, experiment=exp)
        print(f'loss is {epoch_avg_loss}')

        eval_dict = evaluate(model, device, test_loader, evaluator)
        exp.log_figure()
        if 'rocauc' in eval_dict:
            metric = 'rocauc'
        elif 'acc' in eval_dict:
            metric = 'acc'

        rocauc = eval_dict[metric]
        if rocauc > score_needed:
            break
        print(f'Evaluating epoch {epoch}...{metric}: {eval_dict}')
        if exp:
            exp.log_metric('score', rocauc)
    assert rocauc > score_needed, 'could not overfit'
