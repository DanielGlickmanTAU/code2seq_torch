from data.ast_conversion import config, TPE


def count_pairs_efficient(g, counter,i):
    for n in g:
        node = g[n]
        if 'children' not in node:
            continue
        for j in node['children']:
            child_node = g[j]
            if 'type' in child_node and 'type' in node:
                counter[(node['type'], child_node['type'])].append((i, n, j))
                # print(f'something strange.. value in both {g.nodes[n]} and {g.nodes[child]}')


def merge_nodes_efficient(g, parent: int, child: int):
    def merge_children(parent_node, child_node):
        if child in parent_node['children']:
            new_children = [x for x in parent_node['children'] if x != child]
            new_children += child_node['children']
            # parent_node['children'] += child_node['children']
            parent_node['children'] = new_children
        else:
            print(f'bug! {parent_node}:{parent} , {child_node}:{child}')

    parent_node = g[parent]
    child_node = g[child]
    merge_children(parent_node, child_node)
    # if doing efficent counting,implement it about here....
    # todo check cycles
    parent_node['type'] = TPE.order_agnostic_name_merge(parent_node['type'], child_node['type'])

    g.pop(child)
