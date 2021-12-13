from data.ast_conversion import config, TPE


def count_pairs_efficient(g, counter):
    for n in g:
        node = g[n]
        for j in node['children']:
            child_node = g[j]
            if config.skip_if_both_nodes_have_value and 'value' in node and 'value' in child_node:
                pass
                # print(f'something strange.. value in both {g.nodes[n]} and {g.nodes[child]}')
            # if not child_node['children']:
            #     pass
            else:
                counter[(node['type'], child_node['type'])].append((g, n, j))


def merge_nodes_efficient(g, parent: int, child: int):
    def merge_children(parent_node, child_node):
        if child in parent_node['children']:
            parent_node['children'].remove(child)
            parent_node['children'] += child_node['children']
        else:
            print(f'bug! {parent_node}:{parent} , {child_node}:{child}')

    parent_node = g[parent]
    child_node = g[child]
    merge_children(parent_node, child_node)
    # if doing efficent counting,implement it about here....
    # todo check cycles
    parent_node['type'] = TPE.order_agnostic_name_merge(parent_node['type'], child_node['type'])

    g.pop(child)
