#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import io
import tempfile
from typing import Any
from unittest import TestCase, main

import h5py
import numpy as np
import torch

from torchbiggraph.checkpoint_storage import DatasetIO, TwoWayMapping


class TestDatasetIO(TestCase):

    # DatasetIO is only used wrapped in a BufferedReader as a source for
    # torch.load, hence we test it only in this setting.

    @staticmethod
    def save_to(hf: h5py.File, name: str, data: Any) -> None:
        with io.BytesIO() as bf:
            torch.save(data, bf)
            hf.create_dataset(
                name, data=np.frombuffer(bf.getbuffer(), dtype=np.dtype("V1")))

    @staticmethod
    def load_from(hf: h5py.File, name: str) -> Any:
        with io.BufferedReader(DatasetIO(hf[name])) as bf:
            return torch.load(bf)

    def test_scalars(self):
        data = (["a", b"b"], {1: True, 0.2: {None, 4j}})
        # FIXME h5py-2.9 accepts just File(bf), allowing an un-Named TemporaryFile.
        with tempfile.NamedTemporaryFile() as bf:
            with h5py.File(bf.name, "w") as hf:
                self.save_to(hf, "foo", data)
            with h5py.File(bf.name, "r") as hf:
                self.assertEqual(self.load_from(hf, "foo"), data)

    def test_tensors(self):
        data_foo = torch.zeros((100,), dtype=torch.int8)
        data_bar = torch.ones((10, 10))
        # FIXME h5py-2.9 accepts just File(bf), allowing an un-Named TemporaryFile.
        with tempfile.NamedTemporaryFile() as bf:
            with h5py.File(bf.name, "w") as hf:
                self.save_to(hf, "foo", data_foo)
                self.save_to(hf, "bar", data_bar)
            with h5py.File(bf.name, "r") as hf:
                self.assertTrue(data_foo.equal(self.load_from(hf, "foo")))
                self.assertTrue(data_bar.equal(self.load_from(hf, "bar")))

    def test_bad_args(self):
        # FIXME h5py-2.9 accepts just File(bf), allowing an un-Named TemporaryFile.
        with tempfile.NamedTemporaryFile() as bf:
            with h5py.File(bf.name, "w") as hf:
                # Scalar array of "V<length>" type as suggested in the h5py doc.
                data = np.void(b"data")
                with self.assertRaises(TypeError):
                    DatasetIO(hf.create_dataset("foo", data=data))
                # One-dimensional array of uint8 type.
                data = np.frombuffer(b"data", dtype=np.uint8)
                with self.assertRaises(TypeError):
                    DatasetIO(hf.create_dataset("bar", data=data))
                # Two-dimensional array of bytes.
                data = np.frombuffer(b"data", dtype=np.dtype("V1")).reshape(2, 2)
                with self.assertRaises(TypeError):
                    DatasetIO(hf.create_dataset("baz", data=data))


class TestTwoWayMapping(TestCase):

    def test_one_field(self):
        m = TwoWayMapping("foo.bar.{field}", "{field}/ham/eggs", fields=["field"])
        self.assertEqual(m.private_to_public.map("foo.bar.baz"), "baz/ham/eggs")
        self.assertEqual(m.public_to_private.map("spam/ham/eggs"), "foo.bar.spam")
        with self.assertRaises(ValueError):
            m.private_to_public.map("f00.b4r.b4z")
        with self.assertRaises(ValueError):
            m.private_to_public.map("foo.bar")
        with self.assertRaises(ValueError):
            m.private_to_public.map("foo.bar.")
        with self.assertRaises(ValueError):
            m.private_to_public.map("foo.bar.baz.2")
        with self.assertRaises(ValueError):
            m.private_to_public.map("2.foo.bar.baz")
        with self.assertRaises(ValueError):
            m.public_to_private.map("sp4m/h4m/3gg5")
        with self.assertRaises(ValueError):
            m.public_to_private.map("ham/eggs")
        with self.assertRaises(ValueError):
            m.public_to_private.map("/ham/eggs")
        with self.assertRaises(ValueError):
            m.public_to_private.map("2/spam/ham/eggs")
        with self.assertRaises(ValueError):
            m.public_to_private.map("spam/ham/eggs/2")

    def test_many_field(self):
        m = TwoWayMapping("fo{field1}.{field2}ar.b{field3}z",
                          "sp{field3}m/{field2}am/egg{field1}",
                          fields=["field1", "field2", "field3"])
        self.assertEqual(m.private_to_public.map("foo.bar.baz"), "spam/bam/eggo")
        self.assertEqual(m.public_to_private.map("spam/ham/eggs"), "fos.har.baz")


if __name__ == '__main__':
    main()
