from code2seq.utils import compute
import torch_geometric
from torch.utils.data import Dataset

import coloring.graph_generation
from coloring.coloring_utils import color_graph, create_stacks, map_tensor_edge_to_networkx_node_ids
import numpy

torch = compute.get_torch()
device = compute.get_device()


class PyramidEdgeColorDataset(Dataset):
    """ creates a dataset where the inputs are random walk probabilities edges of a single pyramid graph
    and the labels are True/False if the edge connects between nodes of the same color"""

    def __init__(self, max_row_size, num_adj_stack):
        graph, positions = coloring.graph_generation.create_pyramid(1, max_row_size)
        color_graph(graph)
        self.graph = graph
        self.positions = positions

        data = torch_geometric.utils.from_networkx(graph)
        stacks = create_stacks(data, num_adj_stack)
        stacks = stacks.permute(1, 2, 0)
        edge_to_node_ids = map_tensor_edge_to_networkx_node_ids(graph, stacks)

        self.dataset = []
        for edge_tensor, list_of_node_ids in edge_to_node_ids.items():
            edge_tensor = torch.tensor(edge_tensor, device=device)
            for node_i, node_j in list_of_node_ids:
                same_color = graph.nodes[node_i]['color'] == graph.nodes[node_j]['color']
                self.dataset.append(
                    (edge_tensor, torch.tensor(same_color, device=device),
                     self.node_index_tuples_to_tensor(node_i, node_j)))

    @staticmethod
    # node_i,node_j are also tuples of location/index e.g (1,0)
    def node_index_tuples_to_tensor(node_i, node_j):
        return torch.tensor((node_i, node_j))

    @staticmethod
    def tensor_to_node_indexes(tensor):
        if tensor.dim() == 3:
            return [PyramidEdgeColorDataset.tensor_to_node_indexes(e) for e in tensor]

        assert tensor.shape == (2, 2)
        return tuple(tensor[0].numpy()), tuple(tensor[-1].numpy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class PyramidNodeColorDataset(Dataset):
    @staticmethod
    def get_random_index_with_value(tensor, value):
        indexes = torch.where(tensor == value)
        return numpy.random.choice(*indexes)

    def __init__(self, max_row_size):
        graph, positions = coloring.graph_generation.create_pyramid(1, max_row_size)
        color_graph(graph)
        node_colors = torch.tensor([attr['color'] for _, attr in graph.nodes(data=True)])
        # self.positions = positions

        data = torch_geometric.utils.from_networkx(graph)
        graph.positions = positions
        data.graph = graph

        red_index = self.get_random_index_with_value(node_colors, 0)
        green_index = self.get_random_index_with_value(node_colors, 1)
        blue_index = self.get_random_index_with_value(node_colors, 2)

        data.y = node_colors
        data.x = torch.zeros_like(data.y)
        data.x[red_index] = 1
        data.x[green_index] = 2
        data.x[blue_index] = 3

        self.dataset = [data]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
