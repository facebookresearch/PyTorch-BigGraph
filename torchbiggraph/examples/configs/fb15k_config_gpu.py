#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="data/FB15k",
        edge_paths=[
            "data/FB15k/freebase_mtr100_mte100-train_partitioned",
            "data/FB15k/freebase_mtr100_mte100-valid_partitioned",
            "data/FB15k/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="model/fb15k",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=400,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=50,
        batch_size=5000,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        num_gpus=1,
    )

    return config
