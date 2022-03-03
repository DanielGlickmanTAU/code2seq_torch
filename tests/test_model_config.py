from code2seq.utils import compute
import unittest
from unittest import TestCase

from args_parse import get_default_args
from model.model_utils import get_model


class Test(TestCase):
    "comparing with" \
    "https://arxiv.org/pdf/2003.00982.pdf" \
    " https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/SBMs_node_classification/gin_net.py" \
    "https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/configs/SBMs_node_clustering_GIN_PATTERN_500k.json"

    def test_gnn_size_like_benchmarking_gnn_paper(self):
        args = get_default_args()
        args.num_layer = 16
        args.num_transformer_layers = 0
        args.emb_dim = 124
        args.num_heads = 1

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        self.assertTrue(0.9 * 500 <= k_n_params <= 1.1 * 500)

    def test_gnn_small_size_like_benchmarking_gnn_paper(self):
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 0
        args.emb_dim = 110

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        self.assertTrue(0.9 * 100 <= k_n_params <= 1.1 * 100)

    def test_position_transformer_size_like_4_layer_gnn(self):
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'position'
        args.emb_dim = 60
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        k_params_in_4_layer_gnn = 126.5
        self.assertTrue(0.8 * k_params_in_4_layer_gnn <= k_n_params <= 1.2 * k_params_in_4_layer_gnn)

    def test_content_transformer_size_like_4_layer_gnn(self):
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'content'
        args.emb_dim = 52
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        device = compute.get_device()
        args.gin_conv_mlp_hidden_breath = 1
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        k_params_in_4_layer_gnn = 126.5
        self.assertTrue(0.8 * k_params_in_4_layer_gnn <= k_n_params <= 1.2 * k_params_in_4_layer_gnn)

    def test_content_transformer_uses_distance_when_enabled(self):
        device = compute.get_device()
        args = get_default_args()
        args.num_layer = 4
        args.num_transformer_layers = 4
        args.attention_type = 'content'
        args.emb_dim = 52
        args.transformer_ff_dim = 4 * args.emb_dim
        args.num_heads = 4

        args.num_adj_stacks = 1
        args.use_distance_bias = False
        model = get_model(args, 1, device, task='pattern')
        n_params_without_distance_bias = sum(p.numel() for p in model.parameters())

        args.use_distance_bias = True
        model = get_model(args, 1, device, task='pattern')
        n_params_with_distance_bias = sum(p.numel() for p in model.parameters())

        self.assertGreater(n_params_with_distance_bias, n_params_without_distance_bias)

    def test_gnn_gin_hidden_size_taken_from_param(self):
        args = get_default_args()
        args.num_layer = 16
        args.num_transformer_layers = 0
        args.emb_dim = 124
        args.num_heads = 1

        device = compute.get_device()
        # this is the change from the test above
        args.gin_conv_mlp_hidden_breath = 2
        model = get_model(args, 1, device, task='pattern')
        n_params = sum(p.numel() for p in model.parameters())
        k_n_params = n_params / 1_000
        self.assertTrue(k_n_params > 1.1 * 500)

    def test_position_model_dropout_defaults_to_same_as_overall_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[
                   0].dropout.p == model.gnn_transformer.gnn_node.drop_ratio == args.drop_ratio

    def test_position_model_dropout_can_be_different_than_gnn_dropout(self):
        args = get_default_args()
        args.attention_type = 'position'
        args.num_layer = args.num_transformer_layers = 2
        args.transformer_encoder_dropout = 0.123

        model = get_model(args, 3, compute.get_device(), task='mol')
        assert model.gnn_transformer.transformer.layers[0].dropout.p != model.gnn_transformer.gnn_node.drop_ratio


if __name__ == '__main__':
    unittest.main()
