from tqdm import tqdm
import torch

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train_epoch(model, device, loader, optimizer, task_type):
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
            if "classification" in task_type:
                y = batch.y
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return sum(losses) / len(losses)
