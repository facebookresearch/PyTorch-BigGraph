#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from unittest import TestCase, main

from torchbiggraph.util import BucketOrder, create_partition_pairs


class TestCreatePartitionPairs(TestCase):

    def test_valid(self):
        """Ensure every method produces a valid order (contain all pairs once).

        Even if it may not be the intended order.

        """
        orders = [
            BucketOrder.CHAINED_SYMMETRIC_PAIRS,
            BucketOrder.INSIDE_OUT,
            BucketOrder.OUTSIDE_IN,
            BucketOrder.RANDOM,
        ]
        shapes = [(4, 4), (3, 5), (6, 1), (1, 6), (1, 1)]

        for order in orders:
            for nparts_lhs, nparts_rhs in shapes:
                with self.subTest(order=order, shape=(nparts_lhs, nparts_rhs)):
                    actual_pairs_tensor = create_partition_pairs(
                        nparts_lhs=nparts_lhs,
                        nparts_rhs=nparts_rhs,
                        bucket_order=order,
                    ).int()

                    self.assertCountEqual(
                        (tuple(pair) for pair in actual_pairs_tensor.tolist()),
                        product(range(nparts_lhs), range(nparts_rhs))
                    )


if __name__ == '__main__':
    main()
