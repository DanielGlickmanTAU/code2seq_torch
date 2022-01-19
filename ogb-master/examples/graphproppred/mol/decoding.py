import torch
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig


class LSTMDecoder(torch.nn.Module):
    def __init__(self, args, emb_dim, num_tasks, max_seq_len=10):
        super(LSTMDecoder, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.num_tasks = num_tasks
        self.max_seq_len = max_seq_len

        config = DictConfig({'decoder_num_layers': 2,
                             'embedding_size': emb_dim,
                             'decoder_size': emb_dim,
                             'rnn_dropout': self.args.drop_ratio})
        self.output_size = self.num_tasks + 1
        decoder_step = LSTMDecoderStep(config, self.output_size)
        self.decoder = Decoder(decoder_step, output_size=self.output_size, sos_token=self.num_tasks,
                               teacher_forcing=1.0)

    def forward(self, h_node, batched_data):
        segment_sizes = torch.unique_consecutive(batched_data.batch, return_counts=True)[1]
        targets = self._get_targets_with_sos_token_added(batched_data)

        x, att_weights = self.decoder(h_node, segment_sizes, self.output_size, target_sequence=targets)

        # drop first prediction(SOS.first dim).. and sos logits(last dim) of each prediciton after
        # this is done to fix training later
        x = x[1:, :, :-1]
        return x

    def _get_targets_with_sos_token_added(self, batched_data):
        sos_pad = torch.full((batched_data.y_arr.shape[0], 1), self.output_size, device=batched_data.batch.device)
        # returns shape (batch,seq_len)
        return torch.cat((sos_pad, batched_data.y_arr), dim=1).T
