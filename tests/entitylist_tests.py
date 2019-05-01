#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence
from unittest import TestCase, main

import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.entitylist import EntityList


def tensor_list_from_lists(lists: Sequence[Sequence[int]]) -> TensorList:
    offsets = torch.tensor([0] + [len(l) for l in lists], dtype=torch.long).cumsum(dim=0)
    data = torch.cat([torch.tensor(l, dtype=torch.long) for l in lists], dim=0)
    return TensorList(offsets, data)


class TestEntityList(TestCase):

    def test_empty(self):
        self.assertEqual(
            EntityList.empty(),
            EntityList(torch.empty((0,), dtype=torch.long), TensorList.empty()),
        )

    def test_from_tensor(self):
        self.assertEqual(
            EntityList.from_tensor(torch.tensor([3, 4], dtype=torch.long)),
            EntityList(torch.tensor([3, 4], dtype=torch.long), TensorList.empty(num_tensors=2)),
        )

    def test_from_tensor_list(self):
        tensor_list = tensor_list_from_lists([[3, 4], [0, 2]])
        self.assertEqual(
            EntityList.from_tensor_list(tensor_list),
            EntityList(torch.full((2,), -1, dtype=torch.long), tensor_list),
        )

    def test_cat(self):
        tensor_1 = torch.tensor([2, 3], dtype=torch.long)
        tensor_2 = torch.tensor([0, 1], dtype=torch.long)
        tensor_sum = torch.tensor([2, 3, 0, 1], dtype=torch.long)
        tensor_list_1 = tensor_list_from_lists([[3, 4], [0]])
        tensor_list_2 = tensor_list_from_lists([[1, 2, 0], []])
        tensor_list_sum = tensor_list_from_lists([[3, 4], [0], [1, 2, 0], []])
        self.assertEqual(
            EntityList.cat([
                EntityList(tensor_1, tensor_list_1),
                EntityList(tensor_2, tensor_list_2),
            ]),
            EntityList(tensor_sum, tensor_list_sum),
        )

    def test_constructor_checks(self):
        with self.assertRaises(ValueError):
            EntityList(
                torch.tensor([3, 4, 0], dtype=torch.long),
                tensor_list_from_lists([[2, 1]]),
            )

    def test_to_tensor(self):
        self.assertTrue(
            torch.equal(
                EntityList(
                    torch.tensor([2, 3], dtype=torch.long),
                    tensor_list_from_lists([[], []]),
                ).to_tensor(),
                torch.tensor([2, 3], dtype=torch.long),
            ),
        )

    def test_to_tensor_list(self):
        self.assertEqual(
            EntityList(
                torch.tensor([-1, -1], dtype=torch.long),
                tensor_list_from_lists([[3, 4], [0]]),
            ).to_tensor_list(),
            tensor_list_from_lists([[3, 4], [0]]),
        )

    def test_equal(self):
        el = EntityList(
            torch.tensor([3, 4], dtype=torch.long),
            tensor_list_from_lists([[], [2, 1, 0]]),
        )
        self.assertEqual(el, el)
        self.assertNotEqual(
            el,
            EntityList(
                torch.tensor([4, 2], dtype=torch.long),
                tensor_list_from_lists([[], [2, 1, 0]]),
            ),
        )
        self.assertNotEqual(
            el,
            EntityList(
                torch.tensor([3, 4], dtype=torch.long),
                tensor_list_from_lists([[3], [2, 0]]),
            ),
        )

    def test_len(self):
        self.assertEqual(
            len(EntityList(
                torch.tensor([3, 4], dtype=torch.long),
                tensor_list_from_lists([[], [2, 1, 0]]),
            )),
            2,
        )

    def test_getitem_int(self):
        self.assertEqual(
            EntityList(
                torch.tensor([3, 4, 1, 0], dtype=torch.long),
                tensor_list_from_lists([[2, 1], [0], [], [3, 4, 5]]),
            )[-3],
            EntityList(
                torch.tensor([4], dtype=torch.long),
                tensor_list_from_lists([[0]]),
            ),
        )

    def test_getitem_slice(self):
        self.assertEqual(
            EntityList(
                torch.tensor([3, 4, 1, 0], dtype=torch.long),
                tensor_list_from_lists([[2, 1], [0], [], [3, 4, 5]]),
            )[1:3],
            EntityList(
                torch.tensor([4, 1], dtype=torch.long),
                tensor_list_from_lists([[0], []]),
            ),
        )

    def test_getitem_longtensor(self):
        self.assertEqual(
            EntityList(
                torch.tensor([3, 4, 1, 0], dtype=torch.long),
                tensor_list_from_lists([[2, 1], [0], [], [3, 4, 5]]),
            )[torch.tensor([2, 0])],
            EntityList(
                torch.tensor([1, 3], dtype=torch.long),
                tensor_list_from_lists([[], [2, 1]]),
            ),
        )


if __name__ == '__main__':
    main()
