from code2seq.utils import compute  ##import this first
import torch_geometric

from coloring.color_datasets import PyramidNodeColorDataset
from model import model_utils
from args_parse import get_default_args
from data import dataloader_utils
from model.positional.positional_attention_weight import AdjStack
# import train.training
from train import training

torch = compute.get_torch()

### importing OGB
from ogb.graphproppred import Evaluator


def main():
    args = get_default_args()

    torch_geometric.seed_everything(args.seed)

    # pyramid_size = args.pyramid_size
    pyramid_size = 10
    num_adj_stacks = pyramid_size - 1
    dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
    dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

    args = get_default_args()
    # args.num_transformer_layers = 0
    # args.num_layer = 6
    args.drop_ratio = 0.
    args.transformer_encoder_dropout = 0.
    args.emb_dim = 100
    args.num_heads = 1

    num_colors = 3
    device = compute.get_device()
    task = 'coloring'
    model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task, num_embedding=num_colors + 1)
    evaluator = Evaluator('coloring')
    loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args))
    test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=32, mapping=AdjStack(args),
                                                         shuffle=False)
    # exp = None
    # exp = exp_utils.start_exp(args.exp_name, args, model)
    # test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)

    training.full_train_flow(args, device, evaluator, model, test_loader, loader, loader, 'node classification',
                             'acc')


if __name__ == "__main__":
    main()
