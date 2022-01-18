import torch


class LSTMDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, max_seq_len=10):
        super(LSTMDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.decoder = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.linear_layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        pred_list = []

        cn, hn = self.get_initial_hn_cn(x.size(0), x.device)
        for i in range(self.max_seq_len):
            output, (hn, cn) = self.decoder(x.unsqueeze(0), (hn, cn))
            pred_list.append(self.linear_layer(output).squeeze(0))

        # list of len max_seq_len where each entry is a tensor of shape num_graphs(batch_size) X num_tasks(out_dim)
        return pred_list

    def get_initial_hn_cn(self, batch_size, device):
        size_ = (self.decoder.num_layers, batch_size, self.decoder.hidden_size)
        hn, cn = torch.zeros(size_, device=device), torch.zeros(size_, device=device)
        return cn, hn
