#THIS FILE Works with NX graphs
import networkx as nx

from data.ast_conversion import config
from data.ast_conversion.TPE import order_agnostic_name_merge


def merge_nodes(g: nx.DiGraph, parent: int, child: int):
    parent_node = g.nodes[parent]
    child_node = g.nodes[child]

    g.add_edges_from([parent, grand_children] for grand_children in list(g[child]))
    # if doing efficent counting,implement it about here....
    # todo check cycles
    parent_node['type'] = order_agnostic_name_merge(parent_node['type'], child_node['type'])
    g.remove_node(child)


def count_pairs(g: nx.DiGraph, counter):
    for n, nbrs in g.adjacency():
        childs = nbrs.keys()
        parent_node = g.nodes[n]
        for child in childs:
            # consider ignoring childs with no children - terminals
            child_node = g.nodes[child]
            if config.skip_if_both_nodes_have_value and 'value' in parent_node and 'value' in child_node:
                pass
                # print(f'something strange.. value in both {g.nodes[n]} and {g.nodes[child]}')
            if not child_node['children']:
                pass
            else:
                counter[(parent_node['type'], child_node['type'])].append((g, n, child))