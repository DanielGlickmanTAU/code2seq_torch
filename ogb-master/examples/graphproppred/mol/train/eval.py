import torch
from tqdm import tqdm

from ogb.graphproppred import Evaluator
import visualization


def _lexo(n, max_val):
    s = str(n)
    n = n * 10
    while n <= max_val:
        s = '0' + s
        n = n * 10
    return s


def evaluate(model, device, loader, evaluator: Evaluator, epoch=None):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            print(f'skipping training batch {batch}')
            pass
        else:
            with torch.no_grad():
                pred: torch.Tensor = model(batch)

            assert not pred.isnan().any()
            if evaluator.eval_metric == 'smb' or evaluator.eval_metric == 'coloring':
                y_true.append(batch.y.detach().cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if evaluator.name == 'coloring':
        g = batch[0]
        g_pred = pred[:g.num_nodes]
        g_acc = (g_pred.argmax(dim=-1) == g.y).float().mean()
        g_acc = str(round(g_acc.item(), 2))
        if epoch == 1:
            visualization.draw_pyramid(g, 'x', 'input')
            visualization.draw_pyramid(g, 'y', 'gold')
        if epoch is not None:
            visualization.draw_pyramid(g, g_pred, f'epoch {_lexo(epoch, 1000)}. acc:{g_acc}')

    return evaluator.eval(input_dict)
