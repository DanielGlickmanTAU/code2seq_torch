import sys
from main import main
from unittest import TestCase
import unittest
import os


class MyTestCase(TestCase):
    def test_base(self):
        os.chdir('..')
        _append_params()
        # sys.argv.append('nagasaki.merge_attention')
        # sys.argv.append('plus')

        last_train, best_test = main()
        print(last_train, best_test)
        assert 0.19 <= best_test <= 0.2
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


def _append_params():
    sys.argv.append('--cfg')
    sys.argv.append('tests/configs/graph/color-histogram.yaml')
    sys.argv.append('--max_examples')
    sys.argv.append('200')
    sys.argv.append('--row_sizes')
    sys.argv.append('[8,9,10,11]')
    sys.argv.append('--num_rows')
    sys.argv.append('10')
    sys.argv.append('--words_per_row')
    sys.argv.append('10')
    sys.argv.append('--atom_set')
    sys.argv.append('8')
    sys.argv.append('--num_unique_atoms')
    sys.argv.append('1')
    sys.argv.append('--num_unique_colors')
    sys.argv.append('20')
    sys.argv.append('--row_color_mode')
    sys.argv.append('histogram')
    sys.argv.append('nagasaki.steps')
    sys.argv.append('"[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]"')
    sys.argv.append('gt.layer_type')
    sys.argv.append('CustomGatedGCN+Nagasaki')
    sys.argv.append('gt.layers')
    sys.argv.append('3')
    sys.argv.append('gt.n_heads')
    sys.argv.append('2')
    sys.argv.append('gt.n_layers_gnn_only')
    sys.argv.append('1')
    sys.argv.append('posenc_LapPE.enable')
    sys.argv.append('False')
    sys.argv.append('posenc_LapPE.layers')
    sys.argv.append('0')
    sys.argv.append('dataset.node_encoder_name')
    sys.argv.append('TypeDictNode+RWSE')
    sys.argv.append('posenc_SignNet.enable')
    sys.argv.append('False')
    sys.argv.append('posenc_SignNet.post_layers')
    sys.argv.append('2')
    sys.argv.append('posenc_RWSE.enable')
    sys.argv.append('True')
    sys.argv.append('posenc_RWSE.kernel.times_func')
    sys.argv.append('"range(1, 21)"')
    sys.argv.append('posenc_RWSE.model')
    sys.argv.append('Linear')
    sys.argv.append('posenc_RWSE.dim_pe')
    sys.argv.append('24')
    sys.argv.append('posenc_RWSE.raw_norm_type')
    sys.argv.append('BatchNorm')
    sys.argv.append('optim.base_lr')
    sys.argv.append('0.0004')
    sys.argv.append('dataset.only_color')
    sys.argv.append('False')
    sys.argv.append('gt.dim_hidden')
    sys.argv.append('32')
    sys.argv.append('gnn.dim_inner')
    sys.argv.append('32')
    # sys.argv.append('optim.early_stop_patience')
    sys.argv.append('optim.max_epoch')
    sys.argv.append('50')
    sys.argv.append('nagasaki.edge_model_type')
    sys.argv.append('bn-mlp')
    sys.argv.append('nagasaki.edge_reduction')
    sys.argv.append('linear')
    sys.argv.append('nagasaki.ffn_layers')
    sys.argv.append('1')
    sys.argv.append('nagasaki.learn_edges_weight')
    sys.argv.append('True')
    sys.argv.append('gt.dropout')
    sys.argv.append('0.1')
    sys.argv.append('gt.attn_dropout')
    sys.argv.append('0.1')
    sys.argv.append('nagasaki.kernel')
    sys.argv.append('sigmoid')
    sys.argv.append('wandb.use')
    sys.argv.append('False')


if __name__ == '__main__':
    unittest.main()
