from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric

from exp_utils import get_global_exp


def draw_pyg_graph(graph: Union[torch_geometric.data.Data, nx.Graph], to_undirected=True, with_labels=True, text=None):
    if isinstance(graph, nx.Graph):
        graph = torch_geometric.utils.from_networkx(graph)

    nx_graph = torch_geometric.utils.to_networkx(data=graph, to_undirected=to_undirected)
    nx.draw(nx_graph,
            with_labels=with_labels,
            # labels={key: (value['type'] if 'type' in value else value['value']) for key, value in
            #         ast.items()},
            # with_labels=True
            )
    plt.show()


def draw(graph: Union[torch_geometric.data.Data, nx.Graph], color_tensor, color_map=None, to_undirected=True,
         positions=None, with_labels=False, alpha=None, label=None, fig_name=None, force_show=False):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_geometric.utils.to_networkx(graph, to_undirected=to_undirected)
    # if got network predictions,take the argmax
    color_tensor = color_tensor.squeeze()
    if color_tensor.dim() == 2:
        color_tensor = color_tensor.argmax(dim=-1)

    colors = [x.item() for x in color_tensor]
    if color_map:
        colors = [color_map[x] for x in colors]

    # fig, axe = plt.subplots(figsize=(11, 6))
    fig, axe = plt.subplots()
    axe.set_title(label, loc='right')

    nx.draw(graph,
            # dataset.positions,
            node_color=colors,
            pos=positions,
            # edge_color=edge_colors,
            with_labels=with_labels and alpha is None,
            alpha=alpha,
            label=label,
            ax=axe
            )
    exp = get_global_exp()
    if exp:
        exp.log_figure(figure=fig, figure_name=fig_name)
        if force_show:
            plt.show()
        plt.close('all')
    else:
        plt.show()


basic_color_map = ['red', 'green', 'blue', 'pink', 'yellow', 'orange', 'purple', 'brown', 'crimson', 'cyan',
                   'antiquewhite',
                   'bisque', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
                   'darkolivegreen', 'darksalmon', 'firebrick', 'deepskyblue']
#just so we dont get out of index error..
basic_color_map = basic_color_map * 10


# force_show is for debugging. will show plt locally and not only on  comet
def draw_pyramid(data: torch_geometric.data.Data, color_with: Union[str, torch.Tensor], label=None, fig_name=None,
                 force_show=False):
    """gets PyramidNodeColorDataset and colors it..
    uses positions from graph. color_with is either x, y or a tensor of predictions"""
    positions = data.graph.positions
    alpha = None
    if isinstance(color_with, str):
        if color_with == 'x':
            colors = data.x
            color_map = ['gray'] + basic_color_map
        elif color_with == 'y':
            colors = data.y
            color_map = basic_color_map
        else:
            raise Exception(f'only x and y supported as string. got {color_with}')
    else:  # color_with is predictions. tensor shape (n,n_colors)
        colors = color_with.argmax(dim=-1)
        color_map = basic_color_map
        alpha = color_with.softmax(dim=-1).max(dim=-1)[0].tolist()
    draw(data.graph, color_tensor=colors, color_map=color_map, positions=positions, alpha=alpha, label=label,
         with_labels=True, fig_name=fig_name, force_show=force_show)


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
