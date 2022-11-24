import sys
from main import main
from unittest import TestCase
import unittest
import os


class MyTestCase(TestCase):
    def test_gnn_trans(self):
        os.chdir('..')
        _append_params({'gt.layers': 4, 'gt.n_layers_gnn_only': 2})

        last_train, best_test = main()
        print(last_train, best_test)
        assert 0.22 <= last_train <= 0.24
        assert 0.24 <= best_test <= 0.27

    def test_base(self):
        os.chdir('..')
        _append_params()

        last_train, best_test = main()
        print(last_train, best_test)
        assert 0.23 <= best_test <= 0.25
        assert 0.27 <= last_train <= 0.28

    def test_cls(self):
        os.chdir('..')
        _append_params()

        sys.argv.append('nagasaki.add_cls')
        sys.argv.append('True')

        last_train, best_test = main()
        print(last_train, best_test)
        assert 0.19 <= best_test <= 0.2
        assert 0.27 <= last_train <= 0.28


def _append_params(params: dict = None):
    d = base_dict()
    if params:
        d.update(params)
    for key, value in d.items():
        sys.argv.append(key)
        sys.argv.append(str(value))


def base_dict():
    l = []
    l.append('--cfg')
    l.append('tests/configs/graph/color-histogram.yaml')
    l.append('--max_examples')
    l.append('200')
    l.append('--row_sizes')
    l.append('[8,9,10,11]')
    l.append('--num_rows')
    l.append('10')
    l.append('--words_per_row')
    l.append('10')
    l.append('--atom_set')
    l.append('8')
    l.append('--num_unique_atoms')
    l.append('1')
    l.append('--num_unique_colors')
    l.append('20')
    l.append('--row_color_mode')
    l.append('histogram')
    l.append('nagasaki.steps')
    l.append('"[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]"')
    l.append('gt.layer_type')
    l.append('CustomGatedGCN+Nagasaki')
    l.append('gt.layers')
    l.append('3')
    l.append('gt.n_heads')
    l.append('2')
    l.append('gt.n_layers_gnn_only')
    l.append('1')
    l.append('posenc_LapPE.enable')
    l.append('False')
    l.append('posenc_LapPE.layers')
    l.append('0')
    l.append('dataset.node_encoder_name')
    l.append('TypeDictNode+RWSE')
    l.append('posenc_SignNet.enable')
    l.append('False')
    l.append('posenc_SignNet.post_layers')
    l.append('2')
    l.append('posenc_RWSE.enable')
    l.append('True')
    l.append('posenc_RWSE.kernel.times_func')
    l.append('"range(1, 21)"')
    l.append('posenc_RWSE.model')
    l.append('Linear')
    l.append('posenc_RWSE.dim_pe')
    l.append('24')
    l.append('posenc_RWSE.raw_norm_type')
    l.append('BatchNorm')
    l.append('optim.base_lr')
    l.append('0.0004')
    l.append('dataset.only_color')
    l.append('False')
    l.append('gt.dim_hidden')
    l.append('32')
    l.append('gnn.dim_inner')
    l.append('32')
    l.append('optim.max_epoch')
    l.append('50')
    l.append('nagasaki.edge_model_type')
    l.append('bn-mlp')
    l.append('nagasaki.edge_reduction')
    l.append('linear')
    l.append('nagasaki.ffn_layers')
    l.append('1')
    l.append('nagasaki.learn_edges_weight')
    l.append('True')
    l.append('gt.dropout')
    l.append('0.1')
    l.append('gt.attn_dropout')
    l.append('0.1')
    l.append('nagasaki.kernel')
    l.append('sigmoid')
    l.append('wandb.use')
    l.append('False')
    d = {}
    for i in range(0, len(l), 2):
        d[l[i]] = l[i + 1]
    return d


if __name__ == '__main__':
    unittest.main()
