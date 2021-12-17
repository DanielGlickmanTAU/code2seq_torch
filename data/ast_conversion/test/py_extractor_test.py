import argparse
import json
import unittest

from data.ast_conversion import ast_to_graph
import data.py150k_extractor as py_extractor

parser = argparse.ArgumentParser()
parser.add_argument('--max_path_length', type=int, default=999)
parser.add_argument('--max_path_width', type=int, default=999)
parser.add_argument('--use_method_name', type=bool, default=True)
parser.add_argument('--use_nums', type=bool, default=True)
parser.add_argument('--n_jobs', type=int, default=4)
args = parser.parse_args()


class PyExtractorTest(unittest.TestCase):

    @staticmethod
    def str_to_ast(string):
        return ast_to_graph.with_int_keys(json.loads(string))

    def test_py_extractor_works(self):
        uncompressed_ast = self.str_to_ast(
            """{"60": {"type": "FunctionDef", "children": [61, 62, 71, 99]}, "99": {"type": "decorator_list", "children": []}, "71": {"type": "body", "children": [72, 83]}, "83": {"type": "Return", "children": [84]}, "84": {"type": "Call", "children": [85, 93, 95, 97]}, "97": {"type": "NameLoad", "children": [98]}, "98": {"value": "value", "children": []}, "95": {"type": "NameLoad", "children": [96]}, "96": {"value": "attr", "children": []}, "93": {"type": "NameLoad", "children": [94]}, "94": {"value": "self", "children": []}, "85": {"type": "AttributeLoad", "children": [86, 91]}, "91": {"type": "attr", "children": [92]}, "92": {"value": "__setattr__", "children": []}, "86": {"type": "AttributeLoad", "children": [87, 89]}, "89": {"type": "attr", "children": [90]}, "90": {"value": "local", "children": []}, "87": {"type": "NameLoad", "children": [88]}, "88": {"value": "corolocal", "children": []}, "72": {"type": "Assign", "children": [73, 75]}, "75": {"type": "Call", "children": [76, 81]}, "81": {"type": "NameLoad", "children": [82]}, "82": {"value": "value", "children": []}, "76": {"type": "AttributeLoad", "children": [77, 79]}, "79": {"type": "attr", "children": [80]}, "80": {"value": "ref", "children": []}, "77": {"type": "NameLoad", "children": [78]}, "78": {"value": "weakref", "children": []}, "73": {"type": "NameStore", "children": [74]}, "74": {"value": "value", "children": []}, "62": {"type": "arguments", "children": [63, 70]}, "70": {"type": "defaults", "children": []}, "63": {"type": "args", "children": [64, 66, 68]}, "68": {"type": "NameParam", "children": [69]}, "69": {"value": "value", "children": []}, "66": {"type": "NameParam", "children": [67]}, "67": {"value": "attr", "children": []}, "64": {"type": "NameParam", "children": [65]}, "65": {"value": "self", "children": []}, "61": {"value": "__setattr__", "children": []}}""")
        compressed_ast = self.str_to_ast(
            """{"60": {"type": "FunctionDef@arguments@defaults@decorator_list@args@NameParam@NameParam@NameParam", "children": [61, 71, 65, 67, 69]}, "71": {"type": "body@Assign@NameStore", "children": [83, 75, 74]}, "83": {"type": "Return", "children": [84]}, "84": {"type": "Call@AttributeLoad@attr@NameLoad@NameLoad@NameLoad", "children": [86, 92, 94, 96, 98]}, "98": {"value": "value", "children": []}, "96": {"value": "attr", "children": []}, "94": {"value": "self", "children": []}, "92": {"value": "__setattr__", "children": []}, "86": {"type": "AttributeLoad@attr@NameLoad", "children": [90, 88]}, "90": {"value": "local", "children": []}, "88": {"value": "corolocal", "children": []}, "75": {"type": "Call@AttributeLoad@attr@NameLoad@NameLoad", "children": [80, 82, 78]}, "82": {"value": "value", "children": []}, "80": {"value": "ref", "children": []}, "78": {"value": "weakref", "children": []}, "74": {"value": "value", "children": []}, "69": {"value": "value", "children": []}, "67": {"value": "attr", "children": []}, "65": {"value": "self", "children": []}, "61": {"value": "__setattr__", "children": []}}""")

        paths = py_extractor.collect_all([uncompressed_ast], args, False)
        compressed_paths = py_extractor.collect_all([compressed_ast], args, False)
        assert len(paths) == len(compressed_paths)

    def test_collect_sample(self):
        ast = self.str_to_ast(
            """{"3172": {"type": "FunctionDef@arguments@defaults@decorator_list@args@NameParam@NameParam", "children": [3173, 3181, 3177, 3179]}, "3181": {"type": "body@Expr@Str", "children": [3184]}, "3184": {"value": "An IQ no-op.", "children": []}, "3179": {"value": "iq", "children": []}, "3177": {"value": "self", "children": []}, "3173": {"value": "ignore", "children": []}}""")
        label, *contexts = py_extractor._collect_sample(ast, 3172, args)
        assert len(contexts)


if __name__ == '__main__':
    unittest.main()
