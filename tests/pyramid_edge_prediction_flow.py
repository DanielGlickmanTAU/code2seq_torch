from torch import optim
from torch.utils.data import Dataset

import coloring.graph_generation
from code2seq.utils import compute
from coloring.coloring_utils import color_graph, create_stacks, \
    map_tensor_edge_to_color

torch = compute.get_torch()

from torch.utils.data import DataLoader
import torch_geometric


class PyramidEdgeColorDataset(Dataset):
    """ creates a dataset where the inputs are random walk probabilities edges of a single pyramid graph
    and the labels are True/False if the edge connects between nodes of the same color"""

    def __init__(self, max_row_size, num_adj_stack):
        graph, _ = coloring.graph_generation.create_pyramid(1, max_row_size)
        color_graph(graph)
        self.graph = graph

        data = torch_geometric.utils.from_networkx(graph, all)
        stacks = create_stacks(data, num_adj_stack)
        stacks = stacks.permute(1, 2, 0)
        edge_to_is_same_color = map_tensor_edge_to_color(graph, stacks)

        self.dataset = []
        for edge, same_color_list in edge_to_is_same_color.items():
            edge = torch.tensor(edge)
            for same_color in same_color_list:
                self.dataset.append((edge, torch.tensor(same_color)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


max_row_size = 5
num_adj_stacks = 5
dataset = PyramidEdgeColorDataset(max_row_size, num_adj_stacks)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = torch.nn.Linear(num_adj_stacks, 1)

epoch = 1000
optimizer = optim.Adam(model.parameters(), lr=4e-3)
for i in range(epoch):
    model.train()
    losses = []
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = torch.nn.BCEWithLogitsLoss()(y_hat, y.float().reshape(y_hat.shape))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    correct_all = []
    for step, (x, y) in enumerate(loader):
        y_hat = (model(x).sigmoid() > 0.5).squeeze()
        correct = y_hat == y
        correct_all.extend(correct.tolist())
    acc = sum(correct_all) / len(correct_all)

    print(f' epoch {i}, loss {sum(losses) / len(losses)}, acc:{acc}')

print('a')
