import matplotlib as mpl

from data import drawing
from data.ast_conversion import ast_to_graph

mpl.rcParams['figure.dpi'] = 500
import matplotlib.pyplot as plt
import networkx as nx


class Node:
    def __init__(self, name, children, graph=None):
        self.name = name
        self.merged = []
        self.children = children
        self.graph = graph

    def merge_names(self, child):
        return self.agnostic_name_merge(child)

        j = self.get_child_index(child)
        # need this if child was passed as str
        child = self.children[j]
        m = len(child.children)
        return f'{self.name}-({j} {child.name} {m})'

    def agnostic_name_merge(self, child):
        return f'{self.name}-\n{child.name}'

    def get_child_index(self, child):
        if isinstance(child, str):
            child = [c for c in self.children if c.name == child]
            assert len(child) == 1, 'handle multiple child with same name later'
            child = child[0]
        return self.children.index(child)

    def merge_nodes(self, child):
        self.name = self.merge_names(child)
        j = self.get_child_index(child)
        child = self.children[j]
        # move the m children of c to be the    j,j+1,..j+m-1  children of v
        self.children = self.children[:j] + child.children + self.children[j + 1:]
        del child

    def all_nodes(self):
        yield self
        for child in self.children:
            for c in child.all_nodes():
                yield c

    def size(self):
        return len([x for x in self.all_nodes()])

    def to_adj(self):
        self_adj = [(self.name, child.name) for child in self.children]
        for child in self.children:
            self_adj += child.to_adj()
        return self_adj

    def equals(self, other):
        pass

    def __hash__(self):
        pass

    def to_graph(self):
        g = nx.DiGraph()
        g.add_edges_from(self.to_adj())
        return WrapperGraph(g)

    @staticmethod
    def from_graph_node(name, graph):
        children = [Node.from_graph_node(child, graph) for child in graph[name]]
        return Node(name, children)

    @staticmethod
    def from_graph(graph):
        root = nx.topological_sort(graph).__next__()
        return Node.from_graph_node(root, graph)


def draw_ast(ast):
    nx_graph = ast_to_graph.create_nx_graph(ast)
    plt.figure(figsize=(20, 5))
    pos = drawing.hierarchy_pos(nx_graph)
    nx.draw(nx_graph, pos,
            labels={key: (value['type'] if 'type' in value else value['value']) + str(key) for key, value in ast.items()},
            with_labels=True)
    plt.show()
    # WrapperGraph(nx_graph).draw()


class WrapperGraph(nx.DiGraph):
    def __init__(self, target, **attr):
        super().__init__(**attr)
        self._impl = target

    def __getattr__(self, name):
        # here you may have to exclude thigs; i.e. forward them to
        # self.name instead for self._impl.name
        return getattr(self._impl, name)

    def draw(self):
        # plt.figure(figsize=(20, 14))
        plt.figure(figsize=(20, 5))
        pos = drawing.hierarchy_pos(self._impl)
        nx.draw(self._impl, pos, with_labels=True)
        plt.show()
