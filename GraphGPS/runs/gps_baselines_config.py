def get_gps_laplace_transformer_config():
    return {
        'optim.base_lr': [0.00001, 0.00005],
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [0],
        'posenc_LapPE.model': ['Transformer'],
        'posenc_LapPE.layers': [3]
    }


def get_gnn_transformer_laplace_transformer_config():
    return {
        'optim.base_lr': [0.00001, 0.00005],
        'gt.layers': [12, 8],
        'gt.n_layers_gnn_only': [6],
        'posenc_LapPE.model': ['Transformer'],
        'posenc_LapPE.layers': [3]
    }


def get_signnet_transformer_config():
    return {
        'optim.base_lr': [0.00001, 0.00005],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
    }
