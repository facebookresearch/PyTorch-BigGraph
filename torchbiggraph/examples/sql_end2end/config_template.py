CONFIG_TEMPLATE = """
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.
import torch

def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="{TRAINING_DIR}",
        edge_paths=[
            "{TRAINING_DIR}",
        ],
        checkpoint_path="{MODEL_PATH}",
        # Graph structure
        entities={{
            {ENTITIES_DICT}
        }},
        relations=[
            {RELN_DICT}
        ],
        # Scoring model
        dimension=200,
        comparator="dot",
        # Training
        num_epochs=50,
        num_uniform_negs=1000,
        num_batch_negs=1000,
        batch_size=150_000,
        loss_fn="softmax",
        lr=0.05,
        regularization_coef=1e-3,
        num_gpus=torch.cuda.device_count(),
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config
"""