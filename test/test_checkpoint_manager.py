#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import json
from unittest import TestCase, main

from torchbiggraph.checkpoint_manager import ConfigMetadataProvider, TwoWayMapping
from torchbiggraph.config import ConfigSchema, EntitySchema, RelationSchema


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
        m = TwoWayMapping(
            "fo{field1}.{field2}ar.b{field3}z",
            "sp{field3}m/{field2}am/egg{field1}",
            fields=["field1", "field2", "field3"],
        )
        self.assertEqual(m.private_to_public.map("foo.bar.baz"), "spam/bam/eggo")
        self.assertEqual(m.public_to_private.map("spam/ham/eggs"), "fos.har.baz")


class TestConfigMetadataProvider(TestCase):
    def test_basic(self):
        config = ConfigSchema(
            entities={"e": EntitySchema(num_partitions=1)},
            relations=[RelationSchema(name="r", lhs="e", rhs="e")],
            dimension=1,
            entity_path="foo",
            edge_paths=["bar"],
            checkpoint_path="baz",
            init_entity_path="foo"
        )
        metadata = ConfigMetadataProvider(config).get_checkpoint_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertCountEqual(metadata.keys(), ["config/json"])
        self.assertEqual(
            config, ConfigSchema.from_dict(json.loads(metadata["config/json"]))
        )


if __name__ == "__main__":
    main()
