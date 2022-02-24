import argparse

from arg_parse_utils import bool_
from code2seq.utils import compute
from model.GraphTransformerEncoder import GraphTransformerEncoder
from model.positional.positional_attention_weight import AdjStack

torch = compute.get_torch()
import torch_geometric

def add_args(parser):
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.1)')
    parser.add_argument('--num_layer', type=int, default=6,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--num_transformer_layers', type=int, default=0,
                        help='number of transformer layers after GNN')
    parser.add_argument('--transformer_ff_dim', type=int, default=600,
                        help='transformer feedforward dim')
    # this reads it as list of ints

    parser.add_argument('--graph_pooling', type=str, default='mean',
                        help='graph pooling')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--residual', type=bool_, default=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=15,
                        help='training early stopping patience')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help='will set learning_rate = lr_reduce_factor * learning_rate every lr_schedule_patience epochs that the validation does not decrease')
    parser.add_argument('--lr_schedule_patience', type=int, default=5, help='see lr_reduce_factor')
    parser.add_argument('--num_workers', type=int, default=4 * torch.cuda.device_count(),
                        help='number of workers (default: 0)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--max_graph_dist', type=int, default=20)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--exp_name', type=str, default='graph-filter-network')
    parser.add_argument('--num_heads', type=int, default=4, help='attention heads')
    parser.add_argument('--offline', type=bool_, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--transformer_encoder_dropout', type=float)
    parser.add_argument('--gin_conv_mlp_hidden_breath', type=float, default=1.,
                        help='the size of the hidden dim of gin layers MLP')
    parser.add_argument('--limit_examples', type=int, help='limit dataset sizes, for debugging')


def get_default_args():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    add_args(parser)
    AdjStack.add_args(parser)
    GraphTransformerEncoder.add_args(parser)

    args = parser.parse_args()
    if not args.transformer_encoder_dropout:
        args.transformer_encoder_dropout = args.drop_ratio

    torch_geometric.seed_everything(args.seed)
    return args
