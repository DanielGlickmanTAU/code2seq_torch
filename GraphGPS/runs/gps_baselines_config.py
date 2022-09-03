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
        'gt.n_layers_gnn_only': [6, 10],
        'posenc_LapPE.model': ['Transformer'],
        'posenc_LapPE.layers': [3]
    }


def get_gps_signnet_deepset_config(n_layers=6):
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [n_layers],
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


def get_RWSE_gps_signnet_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [6],
        'gt.n_layers_gnn_only': [0],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet+RWSE',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm"
    }


def get_RWSE_gps_config(n_layers=6):
    return {
        'gt.layers': [n_layers],
        'gt.n_layers_gnn_only': [0],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE',
        'posenc_SignNet.enable': False,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm"
    }


def get_RWSE_GNN_config(layers=10):
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [layers],
        'gt.n_layers_gnn_only': [100],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE',
        'posenc_SignNet.enable': False,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm"
    }


def get_gnn_transformer_signnet_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [12],
        'gt.n_layers_gnn_only': [6, 10],
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
        'gt.n_layers_gnn_only': [6, 10],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+SignNet+RWSE',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm"
    }


def get_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [12],
        # 'gt.n_layers_gnn_only': [6, 10],
        'gt.n_layers_gnn_only': [10],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': 16,
        'posenc_SignNet.layers': 3,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm",

        'dataset.transformer_node_encoder_name': 'SignNet'
    }


def get_STRONG_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config():
    return {
        'optim.base_lr': [0.00003, 0.00005],
        'gt.layers': [12],
        # 'gt.n_layers_gnn_only': [6, 10],
        'gt.n_layers_gnn_only': [10],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE',
        'posenc_SignNet.enable': True,
        'posenc_SignNet.model': 'DeepSet',
        'posenc_SignNet.dim_pe': [32, 64],
        'posenc_SignNet.layers': [3, 6],
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm",

        'dataset.transformer_node_encoder_name': 'SignNet'
    }


def get_nagasaki_config(total_layers=4, gnn_layers=2, far_away=False, rwse=True):
    d = get_nagasaki_basic_config(total_layers=total_layers, gnn_layers=gnn_layers, far_away=far_away)
    d.update({
        # 'optim.base_lr': [0.00003, 0.00005],
        'gt.n_heads': [2],
        # 'gt.n_layers_gnn_only': [6, 10],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE' if rwse else 'TypeDictNode',
        'posenc_SignNet.enable': False,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': rwse,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm",

        # 'dataset.transformer_node_encoder_name': 'SignNet'
    })
    return d


def get_nagasaki_basic_config(total_layers=4, gnn_layers=2, far_away=False):
    return {
        'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18, 19, 20]' if far_away else '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
        'gt.layer_type': 'CustomGatedGCN+Nagasaki' if gnn_layers else 'None+Nagasaki',
        'gt.use_gnn': True if gnn_layers else False,
        'gt.layers': [total_layers],
        'gt.n_layers_gnn_only': [gnn_layers],
    }


def get_content_transformer_config(total_layers=4, gnn_layers=2, far_away=False):
    return {
        # 'optim.base_lr': [0.00003, 0.00005],
        'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18, 19, 20]' if far_away else '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
        'gt.layer_type': 'CustomGatedGCN+Transformer',
        'gt.layers': [total_layers],
        'gt.n_heads': [2],
        # 'gt.n_layers_gnn_only': [6, 10],
        'gt.n_layers_gnn_only': [gnn_layers],
        'posenc_LapPE.enable': [False],
        'posenc_LapPE.layers': [0],
        'dataset.node_encoder_name': 'TypeDictNode+RWSE',
        'posenc_SignNet.enable': False,
        'posenc_SignNet.post_layers': 2,

        'posenc_RWSE.enable': True,
        'posenc_RWSE.kernel.times_func': 'range(1, 21)',
        'posenc_RWSE.model': 'Linear',
        'posenc_RWSE.dim_pe': 24,
        'posenc_RWSE.raw_norm_type': "BatchNorm",

        # 'dataset.transformer_node_encoder_name': 'SignNet'
    }
