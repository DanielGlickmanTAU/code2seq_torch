def get_benchmarking_gnns_hyperparam_search_space(multiple_random_seeds: bool):
    return {
        'residual': [True],
        'learning_rate': [1e-3, 1e-4],
        'gin_conv_mlp_hidden_breath': 1.,
        'lr_schedule_patience': 5,
        'lr_reduce_factor': 0.5,
        'seed': list(range(4)) if multiple_random_seeds else [0]
    }


def get_params_for_position_transformer_search():
    return {
        'attention_type': 'position',
        ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
        'emb_dim': 60,
        'transformer_ff_dim': 4 * 60,
        'num_heads': [1, 4]
    }


def get_params_for_vanilla_transformer_search():
    return {
        'attention_type': 'content',
        ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
        'emb_dim': 52,
        'transformer_ff_dim': 4 * 52,
        'num_heads': [1, 4]
    }
