from torch.utils.data import Dataset

import coloring.graph_generation
from code2seq.utils import compute
from coloring.coloring_utils import color_graph, create_stacks, \
    map_tensor_edge_to_color

torch = compute.get_torch()

import torch_geometric


class PyramidEdgeColorDataset(Dataset):
    def __init__(self, max_row_size, num_adj_stack):
        graph, _ = coloring.graph_generation.create_pyramid(max_row_size, min_row_size)
        color_graph(graph)

        data = torch_geometric.utils.from_networkx(graph, all)
        stacks = create_stacks(data, num_adj_stack)
        stacks = stacks.permute(1, 2, 0)
        edge_to_is_same_color = map_tensor_edge_to_color(graph, stacks)

        self.dataset = []
        for edge, same_color_list in edge_to_is_same_color.items():
            edge = torch.tensor(edge)
            for same_color in same_color_list:
                self.dataset.append((edge, torch.tensor(same_color)))


min_row_size = 1
max_row_size = 6
num_adj_stacks = 3

print('a')
