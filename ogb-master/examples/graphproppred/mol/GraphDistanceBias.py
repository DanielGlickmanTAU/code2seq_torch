import numpy
import torch
from torch import nn

import DistanceCalculator


class GraphDistanceBias(nn.Module):
    def __init__(self, args, num_heads: int, receptive_fields: list = None):
        super(GraphDistanceBias, self).__init__()
        self.args = args
        self.receptive_fields = receptive_fields
        if receptive_fields:
            assert max(receptive_fields) <= DistanceCalculator.unconnected
            assert len(receptive_fields) == num_heads
        self.num_heads = num_heads
        self.max_dist = args.max_graph_dist
        self.far_away_mark = self.max_dist
        self.unreachable_mark = self.max_dist + 1
        # self.max_dist embedding plus
        padding_idx = self.unreachable_mark
        if args.distance_bias:
            self.distance_embedding = nn.Embedding(self.max_dist + 2, num_heads, padding_idx=padding_idx)
            with torch.no_grad():
                self.distance_embedding.weight[padding_idx] = float("-inf")

    "gets numpy distance matrix, where distance to self is 0(i.e entry (i,i) equals 0) and all unreachable entries are np.inf" \
    ":return long tensor where all reachable entries further away than max_dist are turned into max_dist + 1 and unreable are turned into -1"

    def as_fixed_distance_tensor(self, distance_matrix, device):
        # unreachable_index = distance_matrix == numpy.inf
        distance_matrix[distance_matrix > self.max_dist] = self.far_away_mark
        # distance_matrix[unreachable_index] = self.unreachable_mark
        return torch.tensor(distance_matrix, dtype=torch.long, device=device)

    def _embed_distances(self, distances, device):
        distances = torch.tensor(distances, dtype=torch.long, device=device)

        # distance_matrix[unreachable_index] = self.unreachable_mark
        if self.receptive_fields:
            masks = [(distances > i).view((1, distances.size(0), distances.size(1))) for i in self.receptive_fields]
            masks = torch.cat(masks)
        else:
            masks = distances >= DistanceCalculator.unconnected
            masks = torch.cat(self.num_heads * [masks.unsqueeze(0)])
            # x = self.as_fixed_distance_tensor(distances, device)
        distances[distances > self.max_dist] = self.far_away_mark
        x = self.distance_embedding(distances).permute(2, 0, 1)
        x[masks] = float('-inf')

        return x

    def forward(self, batched_data):
        if self.args.distance_bias:
            return [self._embed_distances(x, batched_data.batch.device) for x in
                    batched_data.distances]

        return [None for x in batched_data.distances]
