from typing import Dict

AST = Dict[int, Dict]


def get_first_value(ast, node):
    if 'children' not in node:
        return None
    children_ = node['children']
    if not children_:
        return None
    value_node = ast[children_[0]]
    return value_node['value'] if 'value' in value_node else None


def get_node_values(ast, node):
    if 'children' not in node:
        return None
    children_ = node['children']
    if not children_:
        return None
    return [ast[i]['value'] for i in children_ if 'value' in ast[i]]
