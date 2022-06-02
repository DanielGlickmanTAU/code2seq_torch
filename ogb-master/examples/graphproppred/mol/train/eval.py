import torch
from tqdm import tqdm

from ogb.graphproppred import Evaluator


def evaluate(model, device, loader, evaluator: Evaluator, visualizer=None):
    model.eval()
    y_true = []
    y_pred = []
    graphs = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if batch.x.shape[0] == 1:
            print(f'skipping training batch {batch}')
            pass
        else:
            graphs.extend([batch[i] for i in range(batch.num_graphs)])
            batch = batch.to(device)
            with torch.no_grad():
                pred: torch.Tensor = model(batch)

            assert not pred.isnan().any()
            if evaluator.eval_metric == 'smb' or evaluator.eval_metric == 'coloring':
                y_true.append(batch.y.detach().cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    if visualizer:
        #[(i,batch[i].y.float().var()) for i in range(batch.num_graphs)]
        visualizer(graphs, y_true, y_pred)

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    return evaluator.eval(input_dict)
