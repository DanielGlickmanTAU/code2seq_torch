import torch
from commode_utils.modules import Decoder, LSTMDecoderStep
from omegaconf import DictConfig
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

import decoding
from GNN_transformer import GNNTransformer


class GNN(torch.nn.Module):

    def __init__(self, args, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 num_transformer_layers=0, feed_forward_dim=1024, node_encoder=None):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        num_transformer_layers = num_transformer_layers
        num_layer = num_layer - num_transformer_layers
        assert num_layer >= 0
        assert num_transformer_layers >= 0
        self.args = args
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        self.gnn_transformer = GNNTransformer(JK, args, drop_ratio, emb_dim, feed_forward_dim, gnn_type, num_layer,
                                              num_transformer_layers, residual, virtual_node, node_encoder)

        if node_encoder:
            print('assuming code task')
            self.task = 'code'
            self.max_seq_len = args.max_seq_len
            # self.decoder = decoding.LSTMDecoder(input_dim=emb_dim, output_dim=self.num_tasks, hidden_dim=emb_dim,
            #                                     max_seq_len=self.max_seq_len)
            config = DictConfig({'decoder_num_layers': 2,
                                 'embedding_size': self.emb_dim,
                                 'decoder_size': self.emb_dim,
                                 'rnn_dropout': self.args.drop_ratio})
            decoder_step = LSTMDecoderStep(config, self.num_tasks + 1)
            self.decoder = Decoder(decoder_step, output_size=self.num_tasks + 1, sos_token=self.num_tasks,
                                   teacher_forcing=1.0)
            self.decoder = decoding.LSTMDecoder(args, emb_dim, self.num_tasks, max_seq_len=10)

        else:
            print('assuming mol task')
            self.task = 'mol'
            self.create_pooling(emb_dim)

            if graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_transformer(batched_data)
        # shape (num_graphs, out_dim)

        if self.task == 'mol':
            h_graph = self.pool(h_node, batched_data.batch)
            return self.graph_pred_linear(h_graph)

        return self.decoder(h_node, batched_data)

    def create_pooling(self, emb_dim):
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")


if __name__ == '__main__':
    GNN(num_tasks=10)
