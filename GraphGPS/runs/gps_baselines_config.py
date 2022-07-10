def get_gps_laplace_transformer_config():
    return {
        'optim.base_lr': [0.00001, 0.00003],
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [0],
        'posenc_LapPE.model': ['Transformer'],
        'posenc_LapPE.layers': [3]
    }


def get_gnn_transformer_laplace_transformer_config():
    return {
        'optim.base_lr': [0.00001, 0.00003],
        'gt.layers': [12],
        'gt.n_layers_gnn_only': [6, 9],
        'posenc_LapPE.model': ['Transformer'],
        'posenc_LapPE.layers': [3]
    }


def get_signnet_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [0],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,
    }


def get_gnn_transformer_signnet_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [12],
        'gt.n_layers_gnn_only': [6, 9],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,
    }


def get_RSWE_gnn_transformer_signnet_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [12],
        'gt.n_layers_gnn_only': [6, 9],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet+RWSE',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,
    }
