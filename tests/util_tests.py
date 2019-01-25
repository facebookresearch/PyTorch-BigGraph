#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from unittest import TestCase, main

from torchbiggraph.util import PartitionOrder, create_partition_pairs


class TestCreatePartitionPairs(TestCase):

    def test_valid(self):
        """Ensure every method produces a valid order (contain all pairs once).

        Even if it may not be the intended order.

        """
        orders = [
            PartitionOrder.CHAINED_SYMMETRIC_PAIRS,
            PartitionOrder.INSIDE_OUT,
            PartitionOrder.OUTSIDE_IN,
            PartitionOrder.RANDOM,
        ]
        shapes = [(4, 4), (3, 5), (6, 1), (1, 6), (1, 1)]

        for order in orders:
            for nparts_lhs, nparts_rhs in shapes:
                with self.subTest(order=order, shape=(nparts_lhs, nparts_rhs)):
                    actual_pairs_tensor = create_partition_pairs(
                        nparts_lhs=nparts_lhs,
                        nparts_rhs=nparts_rhs,
                        partition_order=order,
                    ).int()

                    self.assertCountEqual(
                        (tuple(pair) for pair in actual_pairs_tensor.tolist()),
                        product(range(nparts_lhs), range(nparts_rhs))
                    )


if __name__ == '__main__':
    main()
