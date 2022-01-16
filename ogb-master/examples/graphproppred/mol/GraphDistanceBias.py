import numpy
import torch
from torch import nn


class GraphDistanceBias(nn.Module):
    def __init__(self, num_heads: int, max_dist: int = 10):
        super(GraphDistanceBias, self).__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.far_away_mark = max_dist
        self.unreachable_mark = max_dist + 1
        # max_dist embedding plus
        padding_idx = self.unreachable_mark
        self.distance_embedding = nn.Embedding(max_dist + 2, num_heads, padding_idx=padding_idx)
        with torch.no_grad():
            self.distance_embedding.weight[padding_idx] = float("-inf")

    "gets numpy distance matrix, where distance to self is 0(i.e entry (i,i) equals 0) and all unreachable entries are np.inf" \
    ":return long tensor where all reachable entries further away than max_dist are turned into max_dist + 1 and unreable are turned into -1"

    def as_fixed_distance_tensor(self, distance_matrix, device):
        unreachable_index = distance_matrix == numpy.inf
        distance_matrix[distance_matrix > self.max_dist] = self.far_away_mark
        distance_matrix[unreachable_index] = self.unreachable_mark
        return torch.tensor(distance_matrix, dtype=torch.long, device=device)

    def forward(self, batched_data):
        distances_batched = [self.as_fixed_distance_tensor(x, batched_data.batch.device) for x in
                             batched_data.distances]
        distance_embedding = [self.distance_embedding(x).permute(2, 0, 1) for x in distances_batched]
        return distance_embedding
