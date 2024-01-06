#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from itertools import product
from unittest import main, TestCase

from torchbiggraph.train_cpu import IterationManager


class TestIterationManager(TestCase):
    def test_full(self) -> None:
        im = IterationManager(
            num_epochs=2, edge_paths=["A", "B", "C"], num_edge_chunks=4
        )
        self.assertEqual(list(im), list(product(range(2), range(3), range(4))))

    def test_partial(self) -> None:
        im = IterationManager(
            num_epochs=2,
            edge_paths=["A", "B", "C"],
            num_edge_chunks=4,
            iteration_idx=(1 * 3 + 2) * 4 + 3,
        )
        self.assertEqual(list(im), [(1, 2, 3)])

    def test_tampering(self) -> None:
        im = IterationManager(
            num_epochs=2, edge_paths=["A", "B", "C"], num_edge_chunks=4
        )
        it = iter(im)
        self.assertEqual(next(it), (0, 0, 0))
        im.iteration_idx = (0 * 3 + 1) * 4 + 1
        # When calling next it gets incremented.
        self.assertEqual(next(it), (0, 1, 2))
        im.edge_paths = ["foo", "bar"]
        im.num_edge_chunks = 2
        self.assertEqual(next(it), (1, 1, 1))
        im.iteration_idx = 100
        with self.assertRaises(StopIteration):
            next(it)

    def test_properties(self) -> None:
        im = IterationManager(
            num_epochs=2,
            edge_paths=["A", "B", "C"],
            num_edge_chunks=4,
            iteration_idx=(0 * 3 + 1) * 4 + 2,
        )
        self.assertEqual(im.epoch_idx, 0)
        self.assertEqual(im.edge_path_idx, 1)
        self.assertEqual(im.edge_path, "B")
        self.assertEqual(im.edge_chunk_idx, 2)
        self.assertEqual(
            im.get_checkpoint_metadata(),
            {
                "iteration/num_epochs": 2,
                "iteration/epoch_idx": 0,
                "iteration/num_edge_paths": 3,
                "iteration/edge_path_idx": 1,
                "iteration/edge_path": "B",
                "iteration/num_edge_chunks": 4,
                "iteration/edge_chunk_idx": 2,
            },
        )


if __name__ == "__main__":
    main()
