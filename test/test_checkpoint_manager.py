#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import json
from unittest import TestCase, main

from torchbiggraph.checkpoint_manager import ConfigMetadataProvider
from torchbiggraph.config import ConfigSchema, EntitySchema, RelationSchema


class TestConfigMetadataProvider(TestCase):

    def test_basic(self):
        config = ConfigSchema(
            entities={"e": EntitySchema(num_partitions=1)},
            relations=[RelationSchema(name="r", lhs="e", rhs="e")],
            dimension=1,
            entity_path="foo", edge_paths=["bar"], checkpoint_path="baz")
        metadata = ConfigMetadataProvider(config).get_checkpoint_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertCountEqual(metadata.keys(), ["config/json"])
        self.assertEqual(
            config, ConfigSchema.from_dict(json.loads(metadata["config/json"])))


if __name__ == '__main__':
    main()
