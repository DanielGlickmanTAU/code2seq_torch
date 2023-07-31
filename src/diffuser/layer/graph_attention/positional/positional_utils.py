def get_stacks_dim(gd_config):
    steps = gd_config.steps
    if isinstance(steps, str):
        steps = list(eval(steps))
    # +1 because of adding self walk
    return len(steps) + 1


def get_edge_dim(gd_config):
    num_stack = get_stacks_dim(gd_config)
    edge_dim = gd_config.ffn_hidden_multiplier * num_stack
    return edge_dim
