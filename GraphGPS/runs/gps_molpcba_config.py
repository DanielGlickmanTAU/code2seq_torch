def get_hiroshima_config_molpcba():
    return {
        'gt.layers': [10],
        'gt.n_layers_gnn_only': [5, 8],

        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': [16, 32, 64, 128],
        'posenc_SignNet.layers': [3, 6],
        'posenc_SignNet.post_layers': 2,

        'dataset.transformer_node_encoder_name': 'SignNet'
    }


def get_hiroshima_config_ppa():
    return {
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [2, 3, 4],

        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': [16, 32],
        'posenc_SignNet.layers': [3, 4],
        'posenc_SignNet.post_layers': 2,
        'dataset.transformer_node_encoder_name': 'SignNet'
    }


def get_hiroshima_config_ppa_transformer():
    return {
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [2, 3, 4],
        'gt.layer_type': 'CustomGatedGCN+Transformer',

        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': [16, 32],
        'posenc_SignNet.layers': [3, 4],
        'posenc_SignNet.post_layers': 2,
        'dataset.transformer_node_encoder_name': 'SignNet'
    }
