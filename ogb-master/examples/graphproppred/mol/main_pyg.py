from code2seq.utils import compute  ##import this first
import torch_geometric

from args_parse import get_default_args
from data.dataloader_utils import get_train_val_test_loaders
from dataset_transformations import DistanceCalculator
from model.model_utils import get_model
from model.positional.positional_attention_weight import AdjStack

from torchvision import transforms

from train.training import full_train_flow

torch = compute.get_torch()

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def main():
    # Training settings
    args = get_default_args()
    args.dataset = "ogbg-molhiv"

    device = compute.get_device()

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      transform=transforms.Compose([DistanceCalculator(), AdjStack(args)]))

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    test_loader, train_loader, valid_loader = get_train_val_test_loaders(dataset, num_workers=args.num_workers,
                                                                         batch_size=args.batch_size,
                                                                         limit=args.limit_examples)

    model = get_model(args, dataset.num_tasks, device, task='mol')

    full_train_flow(args, device, evaluator, model, test_loader, train_loader, valid_loader,
                    task_type=dataset.task_type, eval_metric=dataset.eval_metric)


if __name__ == "__main__":
    main()
