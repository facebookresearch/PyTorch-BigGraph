#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import random
from itertools import product
from unittest import TestCase, main

from torchbiggraph.bucket_scheduling import create_ordered_buckets
from torchbiggraph.config import BucketOrder


class TestCreateOrderedBuckets(TestCase):
    def test_valid(self):
        """Ensure every method produces a valid order (contain all pairs once).

        Even if it may not be the intended order.

        """
        orders = [
            BucketOrder.RANDOM,
            BucketOrder.AFFINITY,
            BucketOrder.INSIDE_OUT,
            BucketOrder.OUTSIDE_IN,
        ]
        shapes = [(4, 4), (3, 5), (6, 1), (1, 6), (1, 1)]
        generator = random.Random()

        for order in orders:
            for nparts_lhs, nparts_rhs in shapes:
                seed = random.getrandbits(32)
                with self.subTest(
                    order=order, shape=(nparts_lhs, nparts_rhs), seed=seed
                ):
                    generator.seed(seed)
                    actual_buckets = create_ordered_buckets(
                        nparts_lhs=nparts_lhs,
                        nparts_rhs=nparts_rhs,
                        order=order,
                        generator=generator,
                    )

                    self.assertCountEqual(
                        actual_buckets, product(range(nparts_lhs), range(nparts_rhs))
                    )


if __name__ == "__main__":
    main()
