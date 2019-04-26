#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from itertools import zip_longest
from unittest import TestCase, main

import torch

from torchbiggraph.entitylist import EntityList
from torchbiggraph.batching import (
    group_by_relation_type,
    batch_edges_mix_relation_types,
    batch_edges_group_by_relation_type,
)


class TestGroupByRelationType(TestCase):

    def test_basic(self):
        self.assertEqual(
            group_by_relation_type(
                torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long),
                EntityList.from_tensor(torch.tensor(
                    [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor(
                    [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long)),
            ),
            [
                (
                    EntityList.from_tensor(
                        torch.tensor([24, 13, 77, 38], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([75, 9, 49, 64], dtype=torch.long)),
                    0,
                ),
                (
                    EntityList.from_tensor(
                        torch.tensor([93, 31], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([90, 25], dtype=torch.long)),
                    1,
                ),
                (
                    EntityList.from_tensor(
                        torch.tensor([70, 66, 5, 5], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([23, 31, 42, 50], dtype=torch.long)),
                    2,
                ),
            ],
        )

    def test_constant(self):
        self.assertEqual(
            group_by_relation_type(
                torch.tensor([3, 3, 3, 3], dtype=torch.long),
                EntityList.from_tensor(torch.tensor(
                    [93, 24, 13, 31], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor(
                    [90, 75, 9, 25], dtype=torch.long)),
            ),
            [
                (
                    EntityList.from_tensor(torch.tensor(
                        [93, 24, 13, 31], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 75, 9, 25], dtype=torch.long)),
                    3,
                ),
            ],
        )

    def test_empty(self):
        self.assertEqual(
            group_by_relation_type(
                torch.empty((0,), dtype=torch.long),
                EntityList.empty(),
                EntityList.empty(),
            ),
            [],
        )


class TestBatchEdgesMixRelationTypes(TestCase):

    def test_basic(self):
        actual_batches = batch_edges_mix_relation_types(
            EntityList.from_tensor(torch.tensor(
                [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long)),
            EntityList.from_tensor(torch.tensor(
                [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long)),
            torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long),
            batch_size=4,
        )
        expected_batches = [
            (
                EntityList.from_tensor(
                    torch.tensor([93, 24, 13, 31], dtype=torch.long)),
                EntityList.from_tensor(
                    torch.tensor([90, 75, 9, 25], dtype=torch.long)),
                torch.tensor([1, 0, 0, 1], dtype=torch.long),
            ),
            (
                EntityList.from_tensor(
                    torch.tensor([70, 66, 77, 38], dtype=torch.long)),
                EntityList.from_tensor(
                    torch.tensor([23, 31, 49, 64], dtype=torch.long)),
                torch.tensor([2, 2, 0, 0], dtype=torch.long),
            ),
            (
                EntityList.from_tensor(
                    torch.tensor([5, 5], dtype=torch.long)),
                EntityList.from_tensor(
                    torch.tensor([42, 50], dtype=torch.long)),
                torch.tensor([2, 2], dtype=torch.long),
            ),
        ]
        # We can't use assertEqual because == between tensors doesn't work.
        for actual_batch, expected_batch \
                in zip_longest(actual_batches, expected_batches):
            a_lhs, a_rhs, a_rel = actual_batch
            e_lhs, e_rhs, e_rel = expected_batch
            self.assertEqual(a_lhs, e_lhs)
            self.assertEqual(a_rhs, e_rhs)
            self.assertTrue(torch.equal(a_rel, e_rel), "%s != %s" % (a_rel, e_rel))


class TestBatchEdgesGroupByType(TestCase):

    def test_basic(self):
        lhs = EntityList.from_tensor(torch.tensor(
            [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long))
        rhs = EntityList.from_tensor(torch.tensor(
            [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long))
        rel = torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long)
        lhs_by_type = defaultdict(list)
        rhs_by_type = defaultdict(list)
        for batch_lhs, batch_rhs, rel_type in batch_edges_group_by_relation_type(
            lhs, rhs, rel, batch_size=3
        ):
            self.assertIsInstance(batch_lhs, EntityList)
            self.assertLessEqual(batch_lhs.size(0), 3)
            lhs_by_type[rel_type].append(batch_lhs)
            self.assertIsInstance(batch_rhs, EntityList)
            self.assertLessEqual(batch_rhs.size(0), 3)
            rhs_by_type[rel_type].append(batch_rhs)
        self.assertCountEqual(lhs_by_type.keys(), [0, 1, 2])
        self.assertCountEqual(rhs_by_type.keys(), [0, 1, 2])
        self.assertEqual(
            EntityList.cat(lhs_by_type[0]),
            EntityList.from_tensor(torch.tensor(
                [24, 13, 77, 38], dtype=torch.long)))
        self.assertEqual(
            EntityList.cat(rhs_by_type[0]),
            EntityList.from_tensor(torch.tensor(
                [75, 9, 49, 64], dtype=torch.long)))
        self.assertEqual(
            EntityList.cat(lhs_by_type[1]),
            EntityList.from_tensor(torch.tensor(
                [93, 31], dtype=torch.long)))
        self.assertEqual(
            EntityList.cat(rhs_by_type[1]),
            EntityList.from_tensor(torch.tensor(
                [90, 25], dtype=torch.long)))
        self.assertEqual(
            EntityList.cat(lhs_by_type[2]),
            EntityList.from_tensor(torch.tensor(
                [70, 66, 5, 5], dtype=torch.long)))
        self.assertEqual(
            EntityList.cat(rhs_by_type[2]),
            EntityList.from_tensor(torch.tensor(
                [23, 31, 42, 50], dtype=torch.long)))


if __name__ == '__main__':
    main()
