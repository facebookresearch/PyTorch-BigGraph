#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entity_base = "data/FB15k"


def get_torchbiggraph_config():

    config = dict(
        entity_path=entity_base,

        num_epochs=50,

        entities={
            'all': {'num_partitions': 1},
        },

        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'complex_diagonal',
            'all_negs': True,
        }],
        dynamic_relations=True,

        edge_paths=[],

        checkpoint_path='model/fb15k',

        dimension=400,
        global_emb=False,
        comparator='dot',
        loss_fn='softmax',
        lr=0.1,

        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config
