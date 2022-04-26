import torch
from tqdm import tqdm

from ogb.graphproppred import Evaluator
import visualization


def evaluate(model, device, loader, evaluator: Evaluator):
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
        # visualization.draw(batch[0],batch[0].y)
        # visualization.draw(batch[0],batch[0].x)
        # visualization.draw(batch[0],pred[:batch[0].num_nodes])

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
