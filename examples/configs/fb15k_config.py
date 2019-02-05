#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entity_base = "data/FB15k"


def getConf():

    config = dict(
        entityPath=entity_base,

        numEpochs=200,

        entities={
            'all': {'numPartitions': 1},
        },

        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'translation'
        }],

        edgePaths=[],

        model='fb15k_dynamic',
        outdir='model/fb15k',

        dimension=100,
        globalEmb=False,
        maxNorm=1,
        margin=0.2,
        metric='dot',
        lr=0.1,
        numUniformNegs=0,
        numBatchNegs=100,
    )

    return config
