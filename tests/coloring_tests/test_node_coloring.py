from code2seq.utils import compute
import exp_utils
from torch import optim
from torch_geometric.loader import DataLoader

from args_parse import get_default_args
from coloring.datasets import PyramidNodeColorDataset
import visualization
from coloring import coloring_utils
from data import dataloader_utils
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from ogb.graphproppred import Evaluator
from tests import test_flow_utils

pyramid_size = 5
num_adj_stacks = pyramid_size + 1
dataset = PyramidNodeColorDataset.create(max_row_size=pyramid_size)
dataset_test = PyramidNodeColorDataset.create(max_row_size=pyramid_size + 1)

index_to_color = coloring_utils.index_to_color

print(dataset)
args = get_default_args()
# args.dataset = "PATTERN"
args.num_transformer_layers = 0
args.num_layer = 4
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.adj_stack = list(range(num_adj_stacks))
# args.num_heads = 1


num_colors = 3
device = compute.get_device()
task = 'coloring'
model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task, num_embedding=num_colors + 1)
print(model)
evaluator = Evaluator('coloring')
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=64, mapping=AdjStack(args))
test_loader = dataloader_utils.create_dataset_loader(dataset_test, batch_size=32, mapping=AdjStack(args), shuffle=False)
# exp = None
exp = exp_utils.start_exp("test", args, model)

test_flow_utils.train_and_assert_overfit(model, loader, evaluator, 'coloring', exp=exp, test_loader=test_loader)
