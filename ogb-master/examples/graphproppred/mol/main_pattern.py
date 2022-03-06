import consts
from code2seq.utils import compute  ##import this first
import torch_geometric

from args_parse import get_default_args
from data import dataloader_utils
from data.dataloader_utils import transform_to_one_hot
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack
from train.training import full_train_flow

torch = compute.get_torch()

### importing OGB
from ogb.graphproppred import Evaluator
from torchvision import transforms


def main():
    args = get_default_args()

    torch_geometric.seed_everything(args.seed)

    device = compute.get_device()
    dataset_name = "PATTERN"
    train_loader, valid_loader, test_loader = dataloader_utils.pyg_get_train_val_test_loaders(dataset_name,
                                                                                              limit=args.limit_examples,
                                                                                              num_workers=args.num_workers,
                                                                                              batch_size=args.batch_size,
                                                                                              transform=
                                                                                              transform_to_one_hot,
                                                                                              mapping=AdjStack(args))

    evaluator = Evaluator(dataset_name)
    model = get_model(args, num_tasks=consts.pattern_num_tasks, device=device, task='pattern')

    full_train_flow(args, device, evaluator, model, test_loader, train_loader, valid_loader, 'node classification',
                    'acc')


if __name__ == "__main__":
    main()
