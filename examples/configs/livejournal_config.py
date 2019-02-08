#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entities_base = 'data/livejournal'


def get_torchbiggraph_config():

    config = dict(
        entity_path=entities_base,

        edge_paths=[],
        num_epochs=30,

        entities={
            'user_id': {'num_partitions': 1},
        },

        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none'
        }],
        checkpoint_path='model/livejournal',

        dimension=1024,
        global_emb=False,
        lr=0.001,
        hogwild_delay=2
    )

    return config
