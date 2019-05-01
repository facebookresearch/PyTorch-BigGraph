#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase, main

import torch

from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList


class TestEdgeList(TestCase):

    def test_empty(self):
        self.assertEqual(
            EdgeList.empty(),
            EdgeList(EntityList.empty(),
                     EntityList.empty(),
                     torch.empty((0,), dtype=torch.long)),
        )

    def test_cat_scalar_same(self):
        self.assertEqual(
            EdgeList.cat([
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor(0, dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(torch.tensor([1, 0], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([1, 3], dtype=torch.long)),
                    torch.tensor(0, dtype=torch.long),
                ),
            ]),
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor(0, dtype=torch.long),
            ),
        )

    def test_cat_scalar_different(self):
        self.assertEqual(
            EdgeList.cat([
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor(0, dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(torch.tensor([1, 0], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([1, 3], dtype=torch.long)),
                    torch.tensor(1, dtype=torch.long),
                ),
            ]),
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([0, 0, 1, 1], dtype=torch.long),
            ),
        )

    def test_cat_vector(self):
        self.assertEqual(
            EdgeList.cat([
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor([2, 1], dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(torch.tensor([1, 0], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([1, 3], dtype=torch.long)),
                    torch.tensor([3, 0], dtype=torch.long),
                ),
            ]),
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([2, 1, 3, 0], dtype=torch.long),
            ),
        )

    def test_cat_mixed(self):
        self.assertEqual(
            EdgeList.cat([
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor(1, dtype=torch.long),
                ),
                EdgeList(
                    EntityList.from_tensor(torch.tensor([1, 0], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([1, 3], dtype=torch.long)),
                    torch.tensor([3, 0], dtype=torch.long),
                ),
            ]),
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([1, 1, 3, 0], dtype=torch.long),
            ),
        )

    def test_constructor_checks(self):
        with self.assertRaises(ValueError):
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([2], dtype=torch.long)),
                torch.tensor(1, dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor([1], dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor([[1]], dtype=torch.long),
            )

    def test_has_scalar_relation_type(self):
        self.assertTrue(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor(3, dtype=torch.long),
            ).has_scalar_relation_type(),
        )
        self.assertFalse(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor([2, 0], dtype=torch.long),
            ).has_scalar_relation_type(),
        )

    def test_get_relation_type_as_scalar(self):
        self.assertEqual(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor(3, dtype=torch.long),
            ).get_relation_type_as_scalar(),
            3,
        )

    def test_get_relation_type_as_vector(self):
        self.assertTrue(
            torch.equal(
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor([2, 0], dtype=torch.long),
                ).get_relation_type_as_vector(),
                torch.tensor([2, 0], dtype=torch.long),
            ),
        )

    def test_get_relation_type(self):
        self.assertEqual(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor(3, dtype=torch.long),
            ).get_relation_type(),
            3,
        )
        self.assertTrue(
            torch.equal(
                EdgeList(
                    EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                    EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                    torch.tensor([2, 0], dtype=torch.long),
                ).get_relation_type(),
                torch.tensor([2, 0], dtype=torch.long),
            ),
        )

    def test_equal(self):
        el = EdgeList(
            EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
            EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
            torch.tensor([2, 0], dtype=torch.long),
        )
        self.assertEqual(el, el)
        self.assertNotEqual(
            el,
            EdgeList(
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                torch.tensor([2, 0], dtype=torch.long),
            ),
        )
        self.assertNotEqual(
            el,
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor(1, dtype=torch.long),
            ),
        )

    def test_len(self):
        self.assertEqual(
            len(EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2], dtype=torch.long)),
                torch.tensor([2, 0], dtype=torch.long),
            )),
            2,
        )

    def test_getitem_int(self):
        self.assertEqual(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([1, 1, 3, 0], dtype=torch.long),
            )[-3],
            EdgeList(
                EntityList.from_tensor(torch.tensor([4], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([2], dtype=torch.long)),
                torch.tensor(1, dtype=torch.long),
            ),
        )

    def test_getitem_slice(self):
        self.assertEqual(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([1, 1, 3, 0], dtype=torch.long),
            )[1:3],
            EdgeList(
                EntityList.from_tensor(torch.tensor([4, 1], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([2, 1], dtype=torch.long)),
                torch.tensor([1, 3], dtype=torch.long),
            ),
        )

    def test_getitem_longtensor(self):
        self.assertEqual(
            EdgeList(
                EntityList.from_tensor(torch.tensor([3, 4, 1, 0], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([0, 2, 1, 3], dtype=torch.long)),
                torch.tensor([1, 1, 3, 0], dtype=torch.long),
            )[torch.tensor([2, 0])],
            EdgeList(
                EntityList.from_tensor(torch.tensor([1, 3], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([1, 0], dtype=torch.long)),
                torch.tensor([3, 1], dtype=torch.long),
            ),
        )


if __name__ == '__main__':
    main()
