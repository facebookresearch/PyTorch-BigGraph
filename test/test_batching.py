#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from collections import defaultdict
from itertools import zip_longest
from unittest import TestCase, main

import torch

from torchbiggraph.batching import (
    batch_edges_group_by_relation_type,
    batch_edges_mix_relation_types,
    group_by_relation_type,
)
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList


class TestGroupByRelationType(TestCase):

    def test_basic(self):
        self.assertEqual(
            group_by_relation_type(
                EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long)),
                    torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long),
                ),
            ),
            [
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([24, 13, 77, 38], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([75, 9, 49, 64], dtype=torch.long)),
                    torch.tensor(0, dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([93, 31], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([90, 25], dtype=torch.long)),
                    torch.tensor(1, dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([70, 66, 5, 5], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([23, 31, 42, 50], dtype=torch.long)),
                    torch.tensor(2, dtype=torch.long),
                ),
            ],
        )

    def test_constant(self):
        self.assertEqual(
            group_by_relation_type(
                EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [93, 24, 13, 31], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 75, 9, 25], dtype=torch.long)),
                    torch.tensor([3, 3, 3, 3], dtype=torch.long),
                ),
            ),
            [
                EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [93, 24, 13, 31], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 75, 9, 25], dtype=torch.long)),
                    torch.tensor(3, dtype=torch.long),
                ),
            ],
        )

    def test_empty(self):
        self.assertEqual(
            group_by_relation_type(
                EdgeList(
                    EntityList.empty(),
                    EntityList.empty(),
                    torch.empty((0,), dtype=torch.long),
                ),
            ),
            [],
        )


class TestBatchEdgesMixRelationTypes(TestCase):

    def test_basic(self):
        self.assertEqual(
            list(batch_edges_mix_relation_types(
                EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long)),
                    torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long),
                ),
                batch_size=4,
            )),
            [
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([93, 24, 13, 31], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([90, 75, 9, 25], dtype=torch.long)),
                    torch.tensor([1, 0, 0, 1], dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([70, 66, 77, 38], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([23, 31, 49, 64], dtype=torch.long)),
                    torch.tensor([2, 2, 0, 0], dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(
                        torch.tensor([5, 5], dtype=torch.long)),
                    EntityList.from_tensor(
                        torch.tensor([42, 50], dtype=torch.long)),
                    torch.tensor([2, 2], dtype=torch.long),
                ),
            ],
        )


class TestBatchEdgesGroupByType(TestCase):

    def test_basic(self):
        edges = EdgeList(
            EntityList.from_tensor(torch.tensor(
                [93, 24, 13, 31, 70, 66, 77, 38, 5, 5], dtype=torch.long)),
            EntityList.from_tensor(torch.tensor(
                [90, 75, 9, 25, 23, 31, 49, 64, 42, 50], dtype=torch.long)),
            torch.tensor([1, 0, 0, 1, 2, 2, 0, 0, 2, 2], dtype=torch.long),
        )
        edges_by_type = defaultdict(list)
        for batch_edges in batch_edges_group_by_relation_type(edges, batch_size=3):
            self.assertIsInstance(batch_edges, EdgeList)
            self.assertLessEqual(len(batch_edges), 3)
            self.assertTrue(batch_edges.has_scalar_relation_type())
            edges_by_type[batch_edges.get_relation_type_as_scalar()].append(batch_edges)
        self.assertEqual(
            {k: EdgeList.cat(v) for k, v in edges_by_type.items()},
            {
                0: EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [24, 13, 77, 38], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [75, 9, 49, 64], dtype=torch.long)),
                    torch.tensor(0, dtype=torch.long),
                ),
                1: EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [93, 31], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [90, 25], dtype=torch.long)),
                    torch.tensor(1, dtype=torch.long),
                ),
                2: EdgeList(
                    EntityList.from_tensor(torch.tensor(
                        [70, 66, 5, 5], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor(
                        [23, 31, 42, 50], dtype=torch.long)),
                    torch.tensor(2, dtype=torch.long),
                ),
            },
        )


if __name__ == '__main__':
    main()
