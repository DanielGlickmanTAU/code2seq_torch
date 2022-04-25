from torch import optim
from torch_geometric.loader import DataLoader

from args_parse import get_default_args
from code2seq.utils import compute
from coloring.datasets import PyramidNodeColorDataset
import visualization
from coloring import coloring_utils
from data import dataloader_utils
from model import model_utils
from model.positional.positional_attention_weight import AdjStack
from train.training import train_epoch

dataset = PyramidNodeColorDataset(max_row_size=5, num_adj_stack=5)

index_to_color = coloring_utils.index_to_color
# visualization.draw(dataset.dataset[0], dataset.dataset[0].y, color_map=index_to_color)
# visualization.draw(dataset.dataset[0], dataset.dataset[0].x,
#                    color_map=coloring_utils.index_to_color_map_with_white(index_to_color))
print(dataset)
args = get_default_args()
# args.dataset = "PATTERN"
args.num_layer = args.num_transformer_layers = 0
args.drop_ratio = 0.
args.transformer_encoder_dropout = 0.
args.emb_dim = 100
args.adj_stack = [0, 1, 2, 3, 4]
# args.num_heads = 1


num_colors = 3
device = compute.get_device()
task = 'coloring'
model = model_utils.get_model(args, num_tasks=num_colors, device=device, task=task, num_embedding=num_colors + 1)
print(model)
loader = dataloader_utils.create_dataset_loader(dataset, batch_size=32, mapping=AdjStack(args))

optimizer = optim.Adam(model.parameters(), lr=2e-4)
for epoch in range(1, 50000 + 1):
    # epoch_avg_loss = train_epoch(model, device, train_loader, optimizer, task_type, experiment=exp)
    epoch_avg_loss = train_epoch(model, device, loader, optimizer, task)
    print(f'loss is {epoch_avg_loss}')

    # eval_dict = evaluate(model, device, train_loader, evaluator)
    #
    # print(f'Evaluating epoch {epoch}...{metric}: {eval_dict}')
    # if exp:
    #     exp.log_metric('score', rocauc)
