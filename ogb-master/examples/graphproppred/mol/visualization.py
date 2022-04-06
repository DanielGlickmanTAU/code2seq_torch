from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric


def draw_pyg_graph(graph: Union[torch_geometric.data.Data, nx.Graph], to_undirected=True, with_labels=True, text=None):
    if isinstance(graph, nx.Graph):
        graph = torch_geometric.utils.from_networkx(graph)

    nx_graph = torch_geometric.utils.to_networkx(data=graph, to_undirected=to_undirected)
    nx.draw(nx_graph,
            with_labels=with_labels,
            node_color=['gray' if y == -1 else 'red' if y == 1 else 'green' if y == 2 else 'blue' for y in
                        graph.y] if hasattr(graph,
                                            'y') else None,
            # labels={key: (value['type'] if 'type' in value else value['value']) for key, value in
            #         ast.items()},
            # with_labels=True
            )
    plt.show()


def show_matrix(stacks, cmap=None, text=None):
    if isinstance(stacks, torch.Tensor):
        stacks = stacks.detach().numpy()
    im = plt.matshow(stacks, cmap=cmap)
    # plt.matshow(stacks)
    if text:
        plt.title(text)

    plt.show()


def get_edges(graph: torch_geometric.data.Data):
    labels_to_edges = {}
    for label in graph.y.unique():
        for label2 in graph.y.unique():
            labels_to_edges[(label.item(), label2.item())] = []
    edges = torch.tensor(graph.adj_stack).permute(1, 2, 0)
    for i, row in enumerate(edges):
        for j, entry in enumerate(row):
            labels_to_edges[(graph.y[i].item(), graph.y[j].item())].append(entry)

    for key in labels_to_edges:
        labels_to_edges[key] = torch.stack(labels_to_edges[key])
    return labels_to_edges

#            edges = get_edges(train_loader.dataset[0])
#             mean_std_by_labels = {key: (len(value), value.mean(0).tolist(), value.std(0).tolist()) for key, value in
#                                   edges.items()}
# p[y neibour | y self] = {key: value[:,1].sum() / (edges[(key[0],1-key[1])][:,1].sum()   + value[:,1].sum()  ) for key,value in edges.items() }
# p[connect | (y_neigbour,y_self) = {key: value[:,1].sum() / ( len( edges[(key[0],1-key[1])]  ) + len(value)  ) for key,value in edges.items() }
