from code2seq.utils.gridsearch import ListArgument


def get_benchmarking_gnns_hyperparam_search_space(multiple_random_seeds: bool):
    return {
        'residual': [True],
        'learning_rate': [1e-3, 1e-4],
        'gin_conv_mlp_hidden_breath': 1.,
        'lr_schedule_patience': 5,
        'lr_reduce_factor': 0.5,
        'seed': list(range(4)) if multiple_random_seeds else [0]
    }


def get_benchmarking_gnns_hyperparam_search_space_alternative():
    return {
        'residual': [True],
        'learning_rate': [5e-4],
        'gin_conv_mlp_hidden_breath': 1.,
        'lr_schedule_patience': 25,
        'lr_reduce_factor': 0.5,
        'seed': [41],
        'epochs': 1_000,
        'batch_size':50
    }


def get_plain_4_gnn_hyperparam_search_space():
    return {
        'num_layer': 4,
        'num_transformer_layers': 0,
        'emb_dim': 110,
        'batch_size': 32,
        'grad_accum_steps': 4
    }


def get_params_for_position_transformer_search():
    return {
        'attention_type': 'position',
        'adj_stacks': ListArgument([0, 1, 2, 3, 4]),
        ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
        'emb_dim': 60,
        'transformer_ff_dim': 4 * 60,
        'num_heads': [1, 4]
    }


def get_params_for_position_transformer_with_large_distance_search():
    return {
        'attention_type': 'position',
        'adj_stacks': ListArgument([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30]),
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


def get_params_for_content_transformer_with_distance_bias_search():
    return {
        'attention_type': 'content',
        'use_distance_bias': True,
        'adj_stacks': ListArgument([0, 1, 2, 3, 4]),
        ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
        'emb_dim': 52,
        'transformer_ff_dim': 4 * 52,
        'num_heads': [1, 4]
    }


def get_params_for_content_transformer_with_large_distance_bias_search():
    return {
        'attention_type': 'content',
        'use_distance_bias': True,
        'adj_stacks': ListArgument([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30]),
        ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
        'emb_dim': 52,
        'transformer_ff_dim': 4 * 52,
        'num_heads': [1, 4]
    }
