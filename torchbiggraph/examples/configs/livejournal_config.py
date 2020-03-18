#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="data/livejournal",
        edge_paths=["data/train_partitioned", "data/test_partitioned"],
        checkpoint_path="model/livejournal",
        # Graph structure
        entities={"user_id": {"num_partitions": 1}},
        relations=[
            {"name": "follow", "lhs": "user_id", "rhs": "user_id", "operator": "none"}
        ],
        # Scoring model
        dimension=1024,
        global_emb=False,
        # Training
        num_epochs=30,
        lr=0.001,
        # Misc
        hogwild_delay=2,
    )

    return config
