import torch
from tqdm import tqdm

from ogb.graphproppred import Evaluator


def evaluate(model, device, loader, evaluator: Evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred: torch.Tensor = model(batch)

            assert not pred.isnan().any()
            if evaluator.eval_metric == 'smb':
                y_true.append(batch.y.detach().cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
