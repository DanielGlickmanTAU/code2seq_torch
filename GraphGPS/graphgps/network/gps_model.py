import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.layer.graph_attention.positional.cls import CLSNode, CLSHead
from graphgps.layer.graph_attention.positional.positional_attention_weight import Diffuser
from graphgps.network.gps_feature_encoder import FeatureEncoder


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
        self.nagasaki_config = cfg.nagasaki
        layers = []
        if n_layers_gnn_only:
            self.local_layers = torch.nn.ModuleList()
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
                                 nagasaki_config=cfg.nagasaki, ffn_multiplier=cfg.gt.ffn_multiplier)
            gps_layer.layer_index = (i, n_gt_layers)
            if i < n_layers_gnn_only:
                self.local_layers.append(gps_layer)
            else:
                layers.append(gps_layer)
        self.layers = torch.nn.Sequential(*layers)
        if cfg.nagasaki.add_cls and not cfg.nagasaki.skip_cls_pooling and cfg.gnn.head != 'inductive_edge' and cfg.dataset.name != 'ogbg-code2':
            self.post_mp = CLSHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out, task=cfg.dataset.task)
        else:
            GNNHead = register.head_dict[cfg.gnn.head]
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        if self.local_layers:
            gnn_outputs = []
            for layer in self.local_layers:
                batch = layer(batch)
                gnn_outputs.append(batch.x)
            if self.nagasaki_config.type == 'vid':
                batch.x = torch.cat(gnn_outputs, dim=-1)
        batch = self.layers(batch)
        batch = self.post_mp(batch)
        return batch


register_network('GPSModel', GPSModel)
