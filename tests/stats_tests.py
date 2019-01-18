#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import attr

from deeplearning.projects.filament2.stats import stats, Stats


@stats
class SampleStats(Stats):

    my_int_metric: int = attr.ib()
    my_float_metric: float = attr.ib()


class TestConfig(TestCase):

    def test_sum(self):
        a = SampleStats(my_int_metric=1, my_float_metric=0.1, count=1)
        b = SampleStats(my_int_metric=2, my_float_metric=0.0, count=2)
        c = SampleStats(my_int_metric=0, my_float_metric=0.2, count=2)
        self.assertEqual(
            SampleStats.sum([a, b, c]),
            SampleStats(
                my_int_metric=3,
                my_float_metric=0.30000000000000004,
                count=5,
            ),
        )

    def test_average(self):
        total = SampleStats(my_int_metric=9, my_float_metric=1.2, count=3)
        self.assertEqual(
            total.average(),
            SampleStats(
                my_int_metric=3,
                my_float_metric=0.39999999999999997,
                count=3,
            )
        )

    def test_str(self):
        self.assertEqual(
            str(SampleStats(my_int_metric=1, my_float_metric=0.2, count=3)),
            "my_int_metric:  1 , my_float_metric:  0.2 , count:  3",
        )
