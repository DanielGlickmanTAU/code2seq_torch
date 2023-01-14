from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch


class ProjectThenApply(nn.Module):
    def __init__(self, module, dim, normalization):
        super().__init__()
        self.normalization = normalization
        self.module = module
        self.project = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.project(x)
        if self.normalization == 'softmax':
            x = x.softmax(-1)
        if self.normalization == 'sigmoid':
            x = x.sigmoid()
        if self.normalization == 'norm':
            x = (x - x.mean()) / (x.var() + 0.0001)

        return self.module(x)


class CNNHyper(nn.Module):
    def __init__(
            self, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1, embedding_type='', normalization=None, project_per_layer=False,
            decode_parts=False, args=None):
        super().__init__()

        self.normalization = normalization
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=1,
                                          batch_first=True) if embedding_type == 'attention' else None

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

        if project_per_layer:
            self.normalization = None

            self.c1_weights = ProjectThenApply(self.c1_weights, hidden_dim, normalization)
            self.c1_bias = ProjectThenApply(self.c1_bias, hidden_dim, normalization)
            self.c2_weights = ProjectThenApply(self.c2_weights, hidden_dim, normalization)
            self.c2_bias = ProjectThenApply(self.c2_bias, hidden_dim, normalization)
            self.l1_weights = ProjectThenApply(self.l1_weights, hidden_dim, normalization)
            self.l1_bias = ProjectThenApply(self.l1_bias, hidden_dim, normalization)
            self.l2_weights = ProjectThenApply(self.l2_weights, hidden_dim, normalization)
            self.l2_bias = ProjectThenApply(self.l2_bias, hidden_dim, normalization)
            self.l3_weights = ProjectThenApply(self.l3_weights, hidden_dim, normalization)
            self.l3_bias = ProjectThenApply(self.l3_bias, hidden_dim, normalization)

        self.decode_parts = decode_parts
        if decode_parts:
            self.num_parts = 10  # num of out projections(c1_weights,c1_bias,c2_weights etc)
            self.project_to_seq = nn.Linear(100, self.num_parts * 100)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_dim,
                                                       dim_feedforward=self.out_dim * 2,
                                                       nhead=2,
                                                       batch_first=True)
            num_layers = args.decoder_layers
            self.decoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        self.args = args

    def forward(self, emd):
        if self.attn:
            attn_output, attn_output_weights = self.attn(emd, emd, emd)
            emd = emd + attn_output

        features = self.mlp(emd)
        if self.normalization == 'softmax':
            features = features.softmax(-1)
        if self.normalization == 'sigmoid':
            features = features.sigmoid()
        if self.normalization == 'norm':
            features = (features - features.mean()) / (features.var() + 0.0001)
        hyper_bs = emd.shape[0]

        if self.decode_parts:
            parts = self.project_to_seq(features).view(features.shape[0], self.num_parts,
                                                       features.shape[1])
            mask = nn.Transformer.generate_square_subsequent_mask(self.num_parts).to(
                features) if self.args.causal_attn_decoder else None
            parts = self.decoder(parts, mask)
            return OrderedDict({
                "conv1.weight": self.c1_weights(parts[:, 0, :]).view(hyper_bs, self.n_kernels, self.in_channels, 5, 5),
                "conv1.bias": self.c1_bias(parts[:, 1, :]).view(hyper_bs, -1),
                "conv2.weight": self.c2_weights(parts[:, 2, :]).view(hyper_bs, 2 * self.n_kernels, self.n_kernels, 5,
                                                                     5),
                "conv2.bias": self.c2_bias(parts[:, 3, :]).view(hyper_bs, -1),
                "fc1.weight": self.l1_weights(parts[:, 4, :]).view(hyper_bs, 120, 2 * self.n_kernels * 5 * 5),
                "fc1.bias": self.l1_bias(parts[:, 5, :]).view(hyper_bs, -1),
                "fc2.weight": self.l2_weights(parts[:, 6, :]).view(hyper_bs, 84, 120),
                "fc2.bias": self.l2_bias(parts[:, 7, :]).view(hyper_bs, -1),
                "fc3.weight": self.l3_weights(parts[:, 8, :]).view(hyper_bs, self.out_dim, 84),
                "fc3.bias": self.l3_bias(parts[:, 9, :]).view(hyper_bs, -1),
            })
        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(hyper_bs, self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(hyper_bs, -1),
            "conv2.weight": self.c2_weights(features).view(hyper_bs, 2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(hyper_bs, -1),
            "fc1.weight": self.l1_weights(features).view(hyper_bs, 120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(hyper_bs, -1),
            "fc2.weight": self.l2_weights(features).view(hyper_bs, 84, 120),
            "fc2.bias": self.l2_bias(features).view(hyper_bs, -1),
            "fc3.weight": self.l3_weights(features).view(hyper_bs, self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(hyper_bs, -1),
        })
        return weights


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, connectivity=0.):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        self.connectivity = connectivity
        # self.scale = nn.Parameter()

    def connect(self, x):
        mat = torch.zeros((x.shape[-1], x.shape[-1]))
        for i in range(mat.shape[0]):
            fill = 1.
            for j in range(i, mat.shape[1]):
                mat[i][j] = fill
                fill = fill * self.connectivity

        return x @ mat.to(x)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.connect(x)
        x = F.relu(self.fc2(x))
        x = self.connect(x)
        x = self.fc3(x)
        return x


class HyperWrapper(nn.Module):
    def __init__(self, hypernetwork, n_nodes, embedding_dim):
        super(HyperWrapper, self).__init__()
        self.hypernetwork = hypernetwork
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

    def forward(self, node_ids):
        ids_tensor = torch.tensor(node_ids, dtype=torch.long, device=next(self.parameters()).device).view(-1)
        emds = self.embeddings(ids_tensor)
        return self.hypernetwork(emds)

    def client_message(self, node_id, weights):
        pass
