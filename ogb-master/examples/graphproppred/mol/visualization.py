from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric

try:
    from exp_utils import get_global_exp
except Exception:
    print('failed importing get_global_exp')
    get_global_exp = lambda: None


def draw_pyg_graph(graph: Union[torch_geometric.data.Data, nx.Graph], to_undirected=True, with_labels=True, text=None):
    if isinstance(graph, nx.Graph):
        graph = torch_geometric.utils.from_networkx(graph)

    nx_graph = torch_geometric.utils.to_networkx(data=graph, to_undirected=to_undirected)
    nx.draw(nx_graph, with_labels=with_labels)
    plt.show()


def draw(graph: Union[torch_geometric.data.Data, nx.Graph], colors, color_map=None, to_undirected=True,
         positions=None, with_labels=False, alpha=None, label=None, fig_name=None, force_show=False, step=None):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_geometric.utils.to_networkx(graph, to_undirected=to_undirected)
    # if got network predictions,take the argmax
    if isinstance(colors, torch.Tensor):
        colors = colors.squeeze()
        if colors.dim() == 2:
            colors = colors.argmax(dim=-1)

        colors = [x.item() for x in colors]
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
        exp.log_figure(figure=fig, figure_name=fig_name, step=step, overwrite=False)
        if force_show:
            plt.show()
        plt.close('all')
    else:
        plt.show()


basic_color_map = ['red', 'green', 'blue', 'pink', 'yellow', 'orange', 'purple', 'brown', 'cyan',
                   'antiquewhite',
                   'bisque', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
                   'darkolivegreen', 'darksalmon', 'firebrick', 'deepskyblue', 'crimson']
# just so we dont get out of index error..
basic_color_map = basic_color_map * 10


# force_show is for debugging. will show plt locally and not only on  comet
def draw_pyramid(data: torch_geometric.data.Data, color_with: Union[str, torch.Tensor], color_mode='argmax', label=None,
                 fig_name=None,
                 force_show=False, step=None, positions=True, with_labels=True):
    """gets PyramidNodeColorDataset and colors it..
    uses positions from graph. color_with is either x, y or a tensor of predictions"""
    if positions == True:
        positions = data.graph.positions
    elif not positions:
        positions = None

    alpha = None
    if isinstance(color_with, str):
        if color_with == 'x':
            assert data.x.squeeze().dim() == 1, f'need dim 1 but got {data.x.dim()}. visualize before inserting to network.'
            colors = data.x
            color_map = ['gray'] + basic_color_map
        elif color_with == 'y':
            colors = data.y
            color_map = basic_color_map
        else:
            raise Exception(f'only x and y supported as string. got {color_with}')
    elif color_mode == 'argmax':  # color_with is predictions. tensor shape (n,n_colors)
        colors = color_with.argmax(dim=-1)
        color_map = basic_color_map
        alpha = color_with.softmax(dim=-1).max(dim=-1)[0].tolist()
    else:
        colors = torch.nn.Linear(color_with.shape[-1], 3)(color_with)
        colors = colors - colors.min()
        colors = colors / colors.max()
        color_map = None
        colors = colors.tolist()
    draw(data.graph, colors=colors, color_map=color_map, positions=positions, alpha=alpha, label=label,
         with_labels=with_labels, fig_name=fig_name, force_show=force_show, step=step)


def show_matrix(stacks, cmap=None, text=None):
    if isinstance(stacks, torch.Tensor):
        stacks = stacks.detach().numpy()
    im = plt.matshow(stacks, cmap=cmap)
    # plt.matshow(stacks)
    if text:
        plt.title(text)

    plt.show()


def draw_attention(graph, source_node, attention_matrix):
    nx_id_to_tensor_index = {x: i for i, x in enumerate(graph.nodes())}
    tensor_id_to_nx_index = {i: x for i, x in enumerate(graph.nodes())}
    assert attention_matrix.dim() == 2 and attention_matrix.shape[0] == attention_matrix.shape[1]

    source_node_tensor_index = nx_id_to_tensor_index[source_node]
    source_node_attention_scores = attention_matrix[source_node_tensor_index]
    source_node_attention_scores_nx_indexed = {tensor_id_to_nx_index[i]: score.item() for i, score in
                                               enumerate(source_node_attention_scores)
                                               # this if ignores padding nodes
                                               if i in tensor_id_to_nx_index}
    heatmap = [source_node_attention_scores_nx_indexed[n] for n in graph.nodes]
    nx.draw(graph, graph.positions, node_color=heatmap, with_labels=True, cmap=plt.cm.Reds)
    nx.draw(graph.subgraph(source_node), graph.positions, node_color=[heatmap[source_node_tensor_index]],
            with_labels=True, font_color='red', cmap=plt.cm.Reds)
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
