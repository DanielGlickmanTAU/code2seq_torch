from torch import nn

from gnn import GNN
from ogb.graphproppred.mol_encoder import AtomEncoder


def get_model(args, num_tasks, device, task):
    if task == 'mol':
        node_encoder = AtomEncoder(args.emb_dim)
    elif task == 'pattern':
        pattern_in_dim = 3
        node_encoder = nn.Embedding(pattern_in_dim, args.emb_dim)

    if args.gnn == 'gin':
        model = GNN(args,JK=args.JK, task=task, gnn_type='gin', num_tasks=num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False,
                    num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual, node_encoder=node_encoder).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(args,JK=args.JK, task=task, gnn_type='gin', num_tasks=num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True,
                    num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual, node_encoder=node_encoder).to(device)
    elif args.gnn == 'gcn':
        model = GNN(args,JK=args.JK, task=task, gnn_type='gcn', num_tasks=num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False,
                    num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual, node_encoder=node_encoder).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(args,JK=args.JK, task=task, gnn_type='gcn', num_tasks=num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True,
                    num_transformer_layers=args.num_transformer_layers,
                    feed_forward_dim=args.transformer_ff_dim, graph_pooling=args.graph_pooling,
                    residual=args.residual, node_encoder=node_encoder).to(device)
    else:
        raise ValueError('Invalid GNN type')
    return model
