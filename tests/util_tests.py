#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase, main

from torchbiggraph.util import (
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
            [slice(0, 7), slice(7, 13), slice(13, 19), slice(19, 25)],
        )

    def test_fewer(self):
        self.assertEqual(
            list(split_almost_equally(23, num_parts=4)),
            [slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 23)],
        )


class TestRoundUpToNearestMultiple(TestCase):

    def test_exact(self):
        self.assertEqual(round_up_to_nearest_multiple(24, 4), 24)

    def test_more(self):
        self.assertEqual(round_up_to_nearest_multiple(25, 4), 28)

    def test_fewer(self):
        self.assertEqual(round_up_to_nearest_multiple(23, 4), 24)


if __name__ == '__main__':
    main()
