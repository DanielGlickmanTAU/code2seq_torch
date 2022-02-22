def get_benchmarking_gnns_hyperparam_search_space(multiple_random_seeds: bool):
    return {
        'learning_rate': [1e-3, 1e-4],
        'gin_conv_mlp_hidden_breath': 1.,
        'lr_schedule_patience': 5,
        'lr_reduce_factor': 0.5,
        'seed': list(range(4)) if multiple_random_seeds else [0]
    }
