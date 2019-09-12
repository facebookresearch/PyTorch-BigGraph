#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import tempfile
from unittest import TestCase, main

import h5py
import numpy as np
import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.graph_storages import FileEdgeAppender


class TestFileEdgeAppender(TestCase):

    def test_tensors(self):
        with tempfile.NamedTemporaryFile() as bf:
            with h5py.File(bf.name, "w") as hf, FileEdgeAppender(hf) as buffered_hf:
                buffered_hf.append_tensor(
                    "foo",
                    torch.tensor([1, 2, 3], dtype=torch.long),
                )
                buffered_hf.append_tensor(
                    "bar",
                    torch.tensor([10, 11], dtype=torch.long),
                )
                buffered_hf.append_tensor(
                    "foo",
                    torch.tensor([4], dtype=torch.long),
                )
                buffered_hf.append_tensor(
                    "foo",
                    torch.tensor([], dtype=torch.long),
                )
                buffered_hf.append_tensor(
                    "bar",
                    torch.arange(12, 1_000_000, dtype=torch.long),
                )
                buffered_hf.append_tensor(
                    "foo",
                    torch.tensor([5, 6], dtype=torch.long),
                )

            with h5py.File(bf.name, "r") as hf:
                np.testing.assert_equal(
                    hf["foo"],
                    np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
                )
                np.testing.assert_equal(
                    hf["bar"],
                    np.arange(10, 1_000_000, dtype=np.int64),
                )

    def test_tensor_list(self):
        with tempfile.NamedTemporaryFile() as bf:
            with h5py.File(bf.name, "w") as hf, FileEdgeAppender(hf) as buffered_hf:
                buffered_hf.append_tensor_list(
                    "foo",
                    TensorList(
                        torch.tensor([0, 3, 5], dtype=torch.long),
                        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                    ),
                )
                buffered_hf.append_tensor_list(
                    "bar",
                    TensorList(
                        torch.tensor([0, 1_000_000], dtype=torch.long),
                        torch.arange(1_000_000, dtype=torch.long),
                    ),
                )
                buffered_hf.append_tensor_list(
                    "foo",
                    TensorList(
                        torch.tensor([0, 1, 1, 3], dtype=torch.long),
                        torch.tensor([6, 7, 8], dtype=torch.long),
                    ),
                )

            with h5py.File(bf.name, "r") as hf:
                np.testing.assert_equal(
                    hf["foo_offsets"],
                    np.array([0, 3, 5, 6, 6, 8], dtype=np.int64),
                )
                np.testing.assert_equal(
                    hf["foo_data"],
                    np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
                )
                np.testing.assert_equal(
                    hf["bar_offsets"],
                    np.array([0, 1_000_000], dtype=np.int64),
                )
                np.testing.assert_equal(
                    hf["bar_data"],
                    np.arange(1_000_000, dtype=np.int64),
                )


if __name__ == '__main__':
    main()
