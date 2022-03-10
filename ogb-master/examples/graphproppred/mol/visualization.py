import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric


def draw_pyg_graph(graph: torch_geometric.data.Data, to_undirected=True):
    nx_graph = torch_geometric.utils.to_networkx(data=graph, to_undirected=to_undirected)
    nx.draw(nx_graph,
            node_color=['red' if y == 1 else 'blue' for y in graph.y],
            # labels={key: (value['type'] if 'type' in value else value['value']) for key, value in
            #         ast.items()},
            # with_labels=True
            )
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
#p[y neibour | y self] = {key: value[:,1].sum() / (edges[(key[0],1-key[1])][:,1].sum()   + value[:,1].sum()  ) for key,value in edges.items() }
#p[connect | (y_neigbour,y_self) = {key: value[:,1].sum() / ( len( edges[(key[0],1-key[1])]  ) + len(value)  ) for key,value in edges.items() }