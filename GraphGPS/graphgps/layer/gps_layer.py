import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
# from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg

from examples.graphproppred.mol.pygraph_utils import to_dense_joined_batch
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE, GINEConvLayer
from graphgps.layer.graph_attention.ContentMultiHeadAttention import ContentMultiheadAttention
from graphgps.layer.graph_attention.positional.nagasaki import Nagasaki, PatternAttention
from graphgps.layer.performer_layer import SelfAttention


class LocalModule(nn.Module):
    def __init__(self, dim_h,
                 local_gnn_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 layer_norm=False, batch_norm=True, gnn_residual=True):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.local_gnn_type = local_gnn_type
        self.residual = gnn_residual

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16,  # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=gnn_residual,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")

        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                if self.residual:
                    h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)

            return h_local


class AttentionLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, nagasaki_config=None, input_stacks=1, cross_stacks=1):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe

        self.nagasaki_config = nagasaki_config
        self.input_stacks = input_stacks

        # Global attention transformer-style model.
        if global_model_type == 'None' or global_model_type is None:
            self.self_attn = None
        elif global_model_type == 'Transformer':
            # self.self_attn = torch.nn.MultiheadAttention(
            #     dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            self.self_attn = ContentMultiheadAttention(dim_h, num_heads, self.attn_dropout, batch_first=True)
        elif global_model_type == 'Nagasaki':
            if cross_stacks > 1:
                self.self_attn = PatternAttention(dim_h, num_heads, self.attn_dropout, nagasaki_config, cross_stacks)
            else:
                self.self_attn = Nagasaki(dim_h, num_heads, self.attn_dropout, nagasaki_config, input_stacks)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, batch, h):
        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_joined_batch(h, batch.batch, self.input_stacks)
            if self.global_model_type == 'Transformer':
                h_attn, att_weights = self.self_attn(h_dense, h_dense, h_dense, attn_mask=~mask, key_padding_mask=None)
                h_attn = h_attn[mask]
            elif self.global_model_type == 'Nagasaki':
                h_attn, att_weights = self.self_attn(batch, h_dense, mask)
                h_attn = h_attn[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")
            # from examples.graphproppred.mol import visualization
            # visualization.draw_attention(batch[index_in_batch].graph, node_id, att_weights[index_in_batch])
            h_attn = self.dropout_attn(h_attn)
            h_attn = h + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            return h_attn


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, nagasaki_config=None, ffn_multiplier=2, gnn_residual=True, input_stacks=1,
                 cross_stacks=1, self_attn_only=False, cross_attn_only=False):
        assert not (cross_stacks > 1 and input_stacks > 1), 'cant do both cross attention and history attention'
        assert not (cross_attn_only and self_attn_only)
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.local_model = None
        self.input_stacks = input_stacks
        if local_gnn_type and local_gnn_type != 'None':
            self.local_model = LocalModule(dim_h, local_gnn_type, num_heads,
                                           pna_degrees=pna_degrees, equivstable_pe=equivstable_pe, dropout=dropout,
                                           layer_norm=layer_norm, batch_norm=batch_norm, gnn_residual=gnn_residual)
        self.local_gnn_type = local_gnn_type
        self.nagasaki_config = nagasaki_config

        # Global attention transformer-style model.
        if global_model_type == 'None' or global_model_type is None:
            self.self_attn = None
        else:
            self.cross_attn = None
            if self_attn_only:
                self.self_attn = AttentionLayer(dim_h,
                                                local_gnn_type, global_model_type, num_heads,
                                                pna_degrees, equivstable_pe, dropout,
                                                attn_dropout, layer_norm, batch_norm,
                                                bigbird_cfg, nagasaki_config, input_stacks, 1)
            elif cross_attn_only:
                self.self_attn = AttentionLayer(dim_h,
                                                local_gnn_type, global_model_type, num_heads,
                                                pna_degrees, equivstable_pe, dropout,
                                                attn_dropout, layer_norm, batch_norm,
                                                bigbird_cfg, nagasaki_config, input_stacks, cross_stacks)
            else:
                self.self_attn = AttentionLayer(dim_h,
                                                local_gnn_type, global_model_type, num_heads,
                                                pna_degrees, equivstable_pe, dropout,
                                                attn_dropout, layer_norm, batch_norm,
                                                bigbird_cfg, nagasaki_config, input_stacks, 1)
                if cross_stacks > 1:
                    self.cross_attn = AttentionLayer(dim_h,
                                                     local_gnn_type, global_model_type, num_heads,
                                                     pna_degrees, equivstable_pe, dropout,
                                                     attn_dropout, layer_norm, batch_norm,
                                                     bigbird_cfg, nagasaki_config, input_stacks, cross_stacks)

        # Feed Forward block.
        if self.self_attn is not None:
            self.activation = F.relu
            self.ff_linear1 = nn.Linear(dim_h, dim_h * ffn_multiplier)
            self.ff_linear2 = nn.Linear(dim_h * ffn_multiplier, dim_h)
            if self.layer_norm:
                # self.norm2 = pygnn.norm.LayerNorm(dim_h)
                self.norm2 = pygnn.norm.GraphNorm(dim_h)
                # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_h)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            h_local = self.local_model(batch)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_attn = self.self_attn(batch, h)
            h_out_list.append(h_attn)

            if self.cross_attn:
                h_attn = self.cross_attn(batch, h)
                h_out_list.append(h_attn)

            # Combine local and global outputs.
            h = sum(h_out_list)

            # Feed Forward block.
            h = h + self._ff_block(h)
            if self.layer_norm:
                h = self.norm2(h, batch.batch)
            if self.batch_norm:
                h = self.norm2(h)
        else:
            assert len(h_out_list) == 1
            h = h_out_list[0]

        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.self_attn}, ' \
            f'heads={self.num_heads}'
        return s
