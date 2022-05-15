#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from unittest import main, TestCase

import torch
from torchbiggraph.util import (
    match_shape,
    round_up_to_nearest_multiple,
    split_almost_equally,
)


class TestSplitAlmostEqually(TestCase):
    def test_exact(self):
        self.assertEqual(
            list(split_almost_equally(24, num_parts=4)),
            [slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 24)],
        )

    def test_more(self):
        self.assertEqual(
            list(split_almost_equally(25, num_parts=4)),
            [slice(0, 7), slice(7, 14), slice(14, 21), slice(21, 25)],
        )

    def test_fewer(self):
        self.assertEqual(
            list(split_almost_equally(23, num_parts=4)),
            [slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 23)],
        )

    def test_so_few_that_last_slice_would_underflow(self):
        # All slices have the same size, which is the ratio size/num_parts
        # rounded up. This however may cause earlier slices to get so many
        # elements that later ones end up being empty. We need to be careful
        # about not returning negative slices in that case.
        self.assertEqual(
            list(split_almost_equally(5, num_parts=4)),
            [slice(0, 2), slice(2, 4), slice(4, 5), slice(5, 5)],
        )
        self.assertEqual(
            list(split_almost_equally(6, num_parts=5)),
            [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 6), slice(6, 6)],
        )


class TestRoundUpToNearestMultiple(TestCase):
    def test_exact(self):
        self.assertEqual(round_up_to_nearest_multiple(24, 4), 24)

    def test_more(self):
        self.assertEqual(round_up_to_nearest_multiple(25, 4), 28)

    def test_fewer(self):
        self.assertEqual(round_up_to_nearest_multiple(23, 4), 24)


class TestMatchShape(TestCase):
    def test_zero_dimensions(self):
        t = torch.zeros(())
        self.assertIsNone(match_shape(t))
        self.assertIsNone(match_shape(t, ...))
        with self.assertRaises(TypeError):
            match_shape(t, 0)
        with self.assertRaises(TypeError):
            match_shape(t, 1)
        with self.assertRaises(TypeError):
            match_shape(t, -1)

    def test_one_dimension(self):
        t = torch.zeros((3,))
        self.assertIsNone(match_shape(t, 3))
        self.assertIsNone(match_shape(t, ...))
        self.assertIsNone(match_shape(t, 3, ...))
        self.assertIsNone(match_shape(t, ..., 3))
        self.assertEqual(match_shape(t, -1), 3)
        with self.assertRaises(TypeError):
            match_shape(t)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 1)
        with self.assertRaises(TypeError):
            match_shape(t, 3, ..., 3)

    def test_many_dimension(self):
        t = torch.zeros((3, 4, 5))
        self.assertIsNone(match_shape(t, 3, 4, 5))
        self.assertIsNone(match_shape(t, ...))
        self.assertIsNone(match_shape(t, ..., 5))
        self.assertIsNone(match_shape(t, 3, ..., 5))
        self.assertIsNone(match_shape(t, 3, 4, 5, ...))
        self.assertEqual(match_shape(t, -1, 4, 5), 3)
        self.assertEqual(match_shape(t, -1, ...), 3)
        self.assertEqual(match_shape(t, -1, 4, ...), 3)
        self.assertEqual(match_shape(t, -1, ..., 5), 3)
        self.assertEqual(match_shape(t, -1, 4, -1), (3, 5))
        self.assertEqual(match_shape(t, ..., -1, -1), (4, 5))
        self.assertEqual(match_shape(t, -1, -1, -1), (3, 4, 5))
        self.assertEqual(match_shape(t, -1, -1, ..., -1), (3, 4, 5))
        with self.assertRaises(TypeError):
            match_shape(t)
        with self.assertRaises(TypeError):
            match_shape(t, 3)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4)
        with self.assertRaises(TypeError):
            match_shape(t, 5, 4, 3)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4, 5, 6)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4, ..., 4, 5)

    def test_bad_args(self):
        t = torch.empty((0,))
        with self.assertRaises(RuntimeError):
            match_shape(t, ..., ...)
        with self.assertRaises(RuntimeError):
            match_shape(t, "foo")
        with self.assertRaises(AttributeError):
            match_shape(None)


if __name__ == "__main__":
    main()
