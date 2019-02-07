#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entity_base = "data/FB15k"


def get_torchbiggraph_conf():

    config = dict(
        entity_path=entity_base,

        num_epochs=200,

        entities={
            'all': {'num_partitions': 1},
        },

        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'translation'
        }],

        edge_paths=[],

        checkpoint_path='model/fb15k',

        dimension=100,
        global_emb=False,
        max_norm=1,
        margin=0.2,
        comparator='dot',
        lr=0.1,
        num_uniform_negs=0,
        num_batch_negs=100,
    )

    return config
