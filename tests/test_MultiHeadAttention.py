from unittest import TestCase
import torch

from graphgps.layer.graph_attention.ContentMultiHeadAttention import ContentAttention
from graphgps.layer.graph_attention.positional.MultiHeadAttention import PositionAttention


class TestMultiHeadAttention(TestCase):
    def test_content_attention_matches_position_attention(self):
        edge_dim = 10
        hidden_dim = 18
        num_heads = 2
        pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=num_heads)
        content_attention = ContentAttention(hidden_dim, num_heads)

        oo = torch.full((edge_dim,), 1.)
        ot = torch.full((edge_dim,), 2.)
        to = torch.full((edge_dim,), 3.)
        tt = torch.full((edge_dim,), 4.)

        stacks = [
                     [oo, oo, ot],
                     [oo, oo, ot],
                     [to, to, tt]], \
                 [
                     [oo, ot, oo],
                     [to, tt, to],
                     [oo, ot, oo]
                 ]
        stack = self.stack_lists_to_tensor(stacks)
        B = 2
        pos_attn_weights = pos_attention(stack, torch.full((B, 3, 3), True))

        one = torch.full((hidden_dim,), 1)
        two = torch.full((hidden_dim,), 2)
        values = torch.stack((torch.stack([one, one, two]), torch.stack([one, two, one]))).float()
        # attention is done batch first
        values = values.transpose(0, 1)
        content_attn_weights = content_attention(values, values, values)[0]

        self.assert_batch_one_attention(content_attn_weights[0])
        self.assert_batch_one_attention(content_attn_weights[1])
        self.assert_batch_two_attention(content_attn_weights[2])
        self.assert_batch_two_attention(content_attn_weights[3])

        self.assert_batch_one_attention(pos_attn_weights[0])
        self.assert_batch_one_attention(pos_attn_weights[1])
        self.assert_batch_two_attention(pos_attn_weights[2])
        self.assert_batch_two_attention(pos_attn_weights[3])

    def assert_batch_one_attention(self, batch_1_head_attention):
        assert batch_1_head_attention[0][0] == batch_1_head_attention[0][1]
        assert batch_1_head_attention[0][0] == batch_1_head_attention[1][1]
        assert batch_1_head_attention[0][0] != batch_1_head_attention[0][2]
        assert batch_1_head_attention[0][2] != batch_1_head_attention[2][2]

    def assert_batch_two_attention(self, batch2_attention):
        assert batch2_attention[0][0] == batch2_attention[0][2]
        assert batch2_attention[0][0] == batch2_attention[2][2]

        assert batch2_attention[0][1] == batch2_attention[2][1]
        assert batch2_attention[1][0] == batch2_attention[1][2]

        assert batch2_attention[0][0] != batch2_attention[0][1]
        assert batch2_attention[0][0] != batch2_attention[1][1]
        assert batch2_attention[1][0] != batch2_attention[1][1]
        assert batch2_attention[2][0] != batch2_attention[2][1]

    def stack_lists_to_tensor(self, stacks):
        return torch.stack([torch.stack([torch.stack(y) for y in x]) for x in (stacks)])

    def test_shaping_position_attention_to_joined_graph_attention(self):
        T = 4
        B = 2
        N = 3
        num_heads = 3

        edge_dim = 10
        project_dim = num_heads * (T ** 2)
        pos_attention = PositionAttention(edge_dim=edge_dim, num_heads=project_dim)

        oo = torch.full((edge_dim,), 1.)
        ot = torch.full((edge_dim,), 2.)
        to = torch.full((edge_dim,), 3.)
        tt = torch.full((edge_dim,), 4.)

        stacks = [
                     [oo, oo, ot],
                     [oo, tt, ot],
                     [to, to, tt]], \
                 [
                     [oo, ot, oo],
                     [to, tt, to],
                     [oo, ot, oo]
                 ]
        stack = self.stack_lists_to_tensor(stacks)

        mask = torch.full((B, 3, 3), True)
        pos_attn_weights = pos_attention(stack, mask)

        pos_attn_weights = PositionAttention.reshape_positional_attention_to_joined_graph_attention(pos_attn_weights, T)

        assert pos_attn_weights.shape == (B * num_heads, N * T, N * T)

        batch1_head1 = 0
        """pos attention should look like:
         <oo_1,oo1>  <oo_1,oo2> ..<oo_1,oo_T> <oo_1, oo1>, <oo_1,oo_2> ..<oo_1,oo_T> <ot_1,ot_1> <ot_1,ot_2> ..<ot_1,ot_t>
         <oo_2,oo1>  <oo_2,oo2> ..<oo_2,oo_T> <oo_2, oo1>, <oo_1,oo_2> ..<oo_2,oo_T> <ot_2,ot_1> <ot_2,ot_2> ..<ot_2,ot_t>
        """
        assert pos_attn_weights[batch1_head1][0][0] != pos_attn_weights[batch1_head1][0][1], \
            'expect <oo_l1,oo_l1> !=<oo_l1,oo_l2>'
        assert pos_attn_weights[batch1_head1][0][0] != pos_attn_weights[batch1_head1][1][0], \
            'expect <oo_l1,oo_l1> != <oo_l2,oo_l1>'

        assert pos_attn_weights[batch1_head1][0][0] == pos_attn_weights[batch1_head1][0][T], \
            'expect <oo_l1,oo_l1> ==<oo_l1,oo_l1>'
        assert pos_attn_weights[batch1_head1][0][0] == pos_attn_weights[batch1_head1][T][0], \
            'expect <oo_l1,oo_l1> ==<oo_l1,oo_l1>'


TestMultiHeadAttention().test_content_attention_matches_position_attention()
