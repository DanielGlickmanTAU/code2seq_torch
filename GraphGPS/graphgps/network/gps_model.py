import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.layer.graph_attention.positional.cls import CLSNode, CLSHead
from graphgps.layer.graph_attention.positional.positional_attention_weight import Diffuser


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in, node_encoder_name=None,
                 edge_encoder_name=None, contract=False):
        super(FeatureEncoder, self).__init__()
        if node_encoder_name is None:
            node_encoder_name = cfg.dataset.node_encoder_name
        if edge_encoder_name is None:
            edge_encoder_name = cfg.dataset.edge_encoder_name
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, expand_x='contract') if contract else NodeEncoder(
                cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder and edge_encoder_name:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if 'n_layers_gnn_only' in cfg.gt:
            n_layers_gnn_only = cfg.gt['n_layers_gnn_only']
        else:
            n_layers_gnn_only = 0

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

        layers = []
        n_gt_layers = cfg.gt.layers
        for i, _ in enumerate(range(n_gt_layers)):
            layer_gnn_type = local_gnn_type
            layer_global_model = global_model_type
            if n_layers_gnn_only > 0:
                # in the first n_layers_gnn_only layers, dont use global model
                if i < n_layers_gnn_only:
                    layer_global_model = 'None'
                else:  # i>= n_layers_gnn_only
                    layer_gnn_type = 'None'
                # done with gnns, starting transformers
            if i == n_layers_gnn_only:
                # encodes between gnns and global model
                if cfg.dataset.transformer_node_encoder_name:
                    layers.append(FeatureEncoder(dim_in, cfg.dataset.transformer_node_encoder_name,
                                                 None, contract=True))

                if global_model_type == 'Nagasaki':
                    layers.append(Diffuser(dim_in, cfg.nagasaki))
                if cfg.nagasaki.add_cls:
                    layers.append(CLSNode(dim_in, cfg.nagasaki))

            gps_layer = GPSLayer(dim_h=cfg.gt.dim_hidden, local_gnn_type=layer_gnn_type,
                                 global_model_type=layer_global_model, num_heads=cfg.gt.n_heads,
                                 pna_degrees=cfg.gt.pna_degrees, equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                                 dropout=cfg.gt.dropout, attn_dropout=cfg.gt.attn_dropout, layer_norm=cfg.gt.layer_norm,
                                 batch_norm=cfg.gt.batch_norm, bigbird_cfg=cfg.gt.bigbird,
                                 nagasaki_config=cfg.nagasaki)
            gps_layer.layer_index = (i, n_gt_layers)
            layers.append(gps_layer)
        self.layers = torch.nn.Sequential(*layers)
        if cfg.nagasaki.add_cls:
            GNNHead = register.head_dict[cfg.gnn.head]
            gnn_head = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
            self.cls_head = CLSHead(gnn_head, cfg.gnn.head)
        else:
            GNNHead = register.head_dict[cfg.gnn.head]
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):

        for module in self.children():
            batch = module(batch)
        return batch


register_network('GPSModel', GPSModel)
