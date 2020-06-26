from typing import List, Any
from unittest import TestCase

import numpy

from configs import PreprocessingConfig
from dataset import BufferedPathContext, Vocabulary
from utils.common import SOS, EOS, PAD, UNK


class TestBufferedPathContext(TestCase):
    def test_store_path_context(self):
        vocab = Vocabulary(
            token_to_id={SOS: 0, EOS: 1, PAD: 2},
            type_to_id={SOS: 1, EOS: 2, PAD: 0},
            label_to_id={SOS: 2, EOS: 0, PAD: 1},
        )
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3)
        labels = [[4], [], [4, 5, 6]]
        from_tokens = [
            [[4], [5, 6]],
            [[], [], []],
            [[6, 5, 4]],
        ]
        path_types = [
            [[4, 5], [6]],
            [[], [], []],
            [[6, 5, 4]],
        ]
        to_tokens = [
            [[6], [4, 5]],
            [[], [], []],
            [[4, 6, 4]],
        ]

        buffered_path_context = BufferedPathContext(config, vocab, labels, from_tokens, path_types, to_tokens)

        true_labels = numpy.array([[2, 2, 2], [4, 0, 4], [0, 1, 5], [1, 1, 6]])
        true_from_tokens = numpy.array([[0, 0, 0, 0, 0, 0], [4, 5, 1, 1, 1, 6], [1, 6, 2, 2, 2, 5], [2, 1, 2, 2, 2, 4]])
        true_path_types = numpy.array([[1, 1, 1, 1, 1, 1], [4, 6, 2, 2, 2, 6], [5, 2, 0, 0, 0, 5], [2, 0, 0, 0, 0, 4]])
        true_to_tokens = numpy.array([[0, 0, 0, 0, 0, 0], [6, 4, 1, 1, 1, 4], [1, 5, 2, 2, 2, 6], [2, 1, 2, 2, 2, 4]])

        self.assertListEqual([2, 3, 1], buffered_path_context.paths_for_label)
        numpy.testing.assert_array_equal(true_labels, buffered_path_context.labels)
        numpy.testing.assert_array_equal(true_from_tokens, buffered_path_context.from_tokens)
        numpy.testing.assert_array_equal(true_path_types, buffered_path_context.path_types)
        numpy.testing.assert_array_equal(true_to_tokens, buffered_path_context.to_tokens)

    def test_store_path_context_check_path_shapes(self):
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3)
        with self.assertRaises(ValueError):
            BufferedPathContext(config, Vocabulary(), [[]], [[], []], [[], [], []], [[]])

    def test_store_path_context_check_full_buffer(self):
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3)
        with self.assertRaises(ValueError):
            BufferedPathContext(config, Vocabulary(), [[], [], []], [[], []], [[], [], []], [[]])

    def test__prepare_to_store_simple(self):
        values = [3, 4, 5]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 1, 2]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))

    def test__prepare_to_store_long(self):
        values = [3, 4, 5, 6, 7, 8, 9, 10]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 6, 7]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))

    def test__prepare_to_store_short(self):
        values = [3]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 1, 2, 2, 2]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))
