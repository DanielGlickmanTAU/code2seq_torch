import collections
import unittest

from data.ast_conversion import TPE, ast_to_graph


class TPETest(unittest.TestCase):

    def test_merge_nodes(self):
        def ast_node(type, children=[]):
            return {'type': type, 'children': children}

        #      X             X
        #      A             AB
        #    B   C   ->    D  E  C
        #   D E
        X_index, A_index, B_index, C_index, D_index, E_index = 0, 1, 2, 3, 4, 5
        graph_index = 12

        ast = {X_index: ast_node('X', [A_index]),
               A_index: ast_node('A', [B_index, C_index]),
               B_index: ast_node('B', [D_index, E_index]),
               C_index: ast_node('C'),
               D_index: ast_node('D'),
               E_index: ast_node('E')
               }

        counter = collections.defaultdict(set, {('X', 'A'): {(graph_index, X_index, A_index)},
                                                ('A', 'B'): {(graph_index, A_index, B_index)},
                                                ('A', 'C'): {(graph_index, A_index, C_index)},
                                                ('B', 'D'): {(graph_index, B_index, D_index)},
                                                ('B', 'E'): {(graph_index, B_index, E_index)}
                                                })
        ast_to_graph.add_parents(ast)
        TPE.merge_nodes_efficient(ast, A_index, B_index, counter, graph_index,max_word_joins=99, from_bottom=False)
        # from defaultdict to dict
        counter = {k: v for k, v in counter.items() if v}

        self.assertDictEqual(counter, {('X', 'A@B'): {(graph_index, X_index, A_index)},
                                       ('A@B', 'D'): {(graph_index, A_index, D_index)},
                                       ('A@B', 'E'): {(graph_index, A_index, E_index)},
                                       ('A@B', 'C'): {(graph_index, A_index, C_index)}
                                       })

    def test_merge_locations(self):
        ast = {
            38: {'type': 'FunctionDef', 'children': [39, 40, 47, 51]},
            51: {'type': 'decorator_list', 'children': []},
            47: {'type': 'body', 'children': [48]},
            48: {'type': 'Return', 'children': [49]},
            49: {'type': 'NameLoad', 'children': [50]},
            50: {'value': 'None', 'children': []},
            40: {'type': 'arguments', 'children': [41, 46]},
            46: {'type': 'defaults', 'children': []},
            41: {'type': 'args', 'children': [42, 44]},
            44: {'type': 'NameParam', 'children': [45]},
            45: {'value': 'request', 'children': []},
            42: {'type': 'NameParam', 'children': [43]},
            43: {'value': 'self', 'children': []},
            39: {'value': 'get_context_data', 'children': []},
        }
        ast_to_graph.add_parents(ast)
        counter = collections.defaultdict(set, {
            ('FunctionDef', 'arguments'): {(0, 38, 40)},
            ('FunctionDef', 'body'): {(0, 38, 47)},
            ('FunctionDef', 'decorator_list'): {(0, 38, 51)},
            ('body', 'Return'): {(0, 47, 48)},
            ('Return', 'NameLoad'): {(0, 48, 49)},
            ('arguments', 'args'): {(0, 40, 41)},
            ('arguments', 'defaults'): {(0, 40, 46)},
            ('args', 'NameParam'): {(0, 41, 44), (0, 41, 42)}
        })

        TPE.merge_locations(counter[('args', 'NameParam')], counter, max_word_joins=99, from_bottom=False, graphs=[ast])


if __name__ == "__main__":
    unittest.main()
