#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from unittest import TestCase, main

from torchbiggraph.distributed import ProcessRanks


class TestProcessRanks(TestCase):

    def test_implicit_partition_servers(self):
        ranks = ProcessRanks.from_num_invocations(3, -1)
        self.assertEqual(ranks.trainers, [0, 1, 2])
        self.assertEqual(ranks.parameter_servers, [3, 4, 5])
        self.assertEqual(ranks.parameter_clients, [6, 7, 8])
        self.assertEqual(ranks.lock_server, 9)
        self.assertEqual(ranks.partition_servers, [10, 11, 12])

    def test_no_partition_servers(self):
        ranks = ProcessRanks.from_num_invocations(4, 0)
        self.assertEqual(ranks.trainers, [0, 1, 2, 3])
        self.assertEqual(ranks.parameter_servers, [4, 5, 6, 7])
        self.assertEqual(ranks.parameter_clients, [8, 9, 10, 11])
        self.assertEqual(ranks.lock_server, 12)
        self.assertEqual(ranks.partition_servers, [])

    def test_explicit_partition_servers(self):
        ranks = ProcessRanks.from_num_invocations(5, 3)
        self.assertEqual(ranks.trainers, [0, 1, 2, 3, 4])
        self.assertEqual(ranks.parameter_servers, [5, 6, 7, 8, 9])
        self.assertEqual(ranks.parameter_clients, [10, 11, 12, 13, 14])
        self.assertEqual(ranks.lock_server, 15)
        self.assertEqual(ranks.partition_servers, [16, 17, 18])


if __name__ == '__main__':
    main()
