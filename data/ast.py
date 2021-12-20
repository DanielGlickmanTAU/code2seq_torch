from typing import Dict

AST = Dict[int, Dict]


def get_node_value(ast, node):
    if 'children' not in node:
        return None
    children_ = node['children']
    if not children_:
        return None
    value_node = ast[children_[0]]
    return value_node['value'] if 'value' in value_node else None