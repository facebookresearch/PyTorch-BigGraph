#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entities_base = 'data/livejournal'


def getConf():

    config = dict(
        entityPath=entities_base,

        edgePaths=[],
        numEpochs=30,

        entities={
            'user_id': {'numPartitions': 1},
        },

        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none'
        }],
        model='livejournal',
        outdir='model/livejournal',

        dimension=1024,
        globalEmb=False,
        lr=0.001,
        hogwild_delay=2
    )

    return config
