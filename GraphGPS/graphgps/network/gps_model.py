import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from examples.graphproppred.mol import pygraph_utils
from examples.graphproppred.mol.pygraph_utils import to_dense_joined_batch
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

        dim_inner = cfg.gnn.dim_inner
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = dim_inner

        assert cfg.gt.dim_hidden == dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        self.nagasaki_config = cfg.nagasaki
        transformer_layers = []
        self.local_layers = []
        if n_layers_gnn_only:
            self.local_layers = torch.nn.ModuleList()
        # layers between gnn and transformers
        self.middle_layers = torch.nn.ModuleList()
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
                    self.middle_layers.append(FeatureEncoder(dim_in, cfg.dataset.transformer_node_encoder_name,
                                                             None, contract=True))

                if global_model_type == 'Nagasaki':
                    self.middle_layers.append(Diffuser(dim_in, cfg.nagasaki))
                if cfg.nagasaki.add_cls:
                    self.middle_layers.append(CLSNode(dim_in, cfg.nagasaki))

            gps_layer = GPSLayer(dim_h=cfg.gt.dim_hidden, local_gnn_type=layer_gnn_type,
                                 global_model_type=layer_global_model, num_heads=cfg.gt.n_heads,
                                 pna_degrees=cfg.gt.pna_degrees, equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                                 dropout=cfg.gt.dropout, attn_dropout=cfg.gt.attn_dropout, layer_norm=cfg.gt.layer_norm,
                                 batch_norm=cfg.gt.batch_norm, bigbird_cfg=cfg.gt.bigbird,
                                 nagasaki_config=cfg.nagasaki, ffn_multiplier=cfg.gt.ffn_multiplier,
                                 gnn_residual=cfg.gnn.residual,
                                 input_stacks=n_layers_gnn_only if self.nagasaki_config.type == 'vid' else 1,
                                 cross_stacks=n_layers_gnn_only if self.nagasaki_config.type == 'cross' else 1)
            gps_layer.layer_index = (i, n_gt_layers)
            if i < n_layers_gnn_only:
                self.local_layers.append(gps_layer)
            else:
                transformer_layers.append(gps_layer)
        self.transformer_layers = torch.nn.Sequential(*transformer_layers)
        self.cls_pool = cfg.nagasaki.add_cls and not cfg.nagasaki.skip_cls_pooling and cfg.gnn.head != 'inductive_edge' and cfg.dataset.name != 'ogbg-code2'
        if self.cls_pool:
            self.post_mp = CLSHead(dim_in=dim_inner, dim_out=dim_out, task=cfg.dataset.task)
        else:
            GNNHead = register.head_dict[cfg.gnn.head]
            self.post_mp = GNNHead(dim_in=dim_inner, dim_out=dim_out)
        self.dim_inner = dim_inner

    def forward(self, batch):
        gnn_outputs = []
        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        if self.nagasaki_config.type == 'cross':
            initial_x = batch.x
        if self.local_layers:
            for layer in self.local_layers:
                batch = layer(batch)
                gnn_outputs.append(batch.x)
        for layer in self.middle_layers:
            batch = layer(batch)

        if self.nagasaki_config.type == 'vid':
            batch.x = pygraph_utils.concat_layer_activations(gnn_outputs)
        if self.nagasaki_config.type == 'cross':
            history = pygraph_utils.concat_layer_activations(gnn_outputs)
            batch.history = to_dense_joined_batch(history, batch.batch, len(gnn_outputs))
            batch.x = initial_x

        if self.nagasaki_config.type.lower() == 'jk':
            batch.x = sum(gnn_outputs)

        batch = self.transformer_layers(batch)
        if self.nagasaki_config.type == 'vid' and not self.cls_pool:
            # pooling works with graph structure, so need batch.x now to have back the original number of nodes..
            # solve this by summing up nodes from different time steps..
            batch.x = batch.x.view(-1, len(gnn_outputs), self.dim_inner).sum(1)
        batch = self.post_mp(batch)
        return batch


register_network('GPSModel', GPSModel)
