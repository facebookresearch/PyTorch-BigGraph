"""
Python Script enlarging existing checkpoint files
"""

import json
import h5py

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.types import SINGLE_TRAINER, Rank
from torchbiggraph.util import (
    setup_logging,
)


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


# TODO: Read the updated count files to initialize new embeddings
# TODO: Can merge this feature to the beginning of the training
# TODO: Load the previous entity mapping and append new entities in new partition
def enlarge_checkpoint_files(config_dict):
    """
    :param:

    :return:
    """
    checkpoint_path = config_dict.checkpoint_path
    (
        entity_configs,
        relation_configs,
        entity_path,
        edge_paths,
        dynamic_relations,
    ) = parse_config_partial(  # noqa
        config_dict
    )

    # Load specified embedding
    embeddings_paths = [checkpoint_path / f"embeddings_{entity_name}_{partition}.v60.h5"
                        for entity_name, entity_config in entity_configs.items()
                        for partition in range(entity_config.num_partitions)
                        ]

    for emb_path in embeddings_paths:
        with h5py.File(emb_path, "r") as hf:
            print(hf["embeddings"][:])


def main():
    setup_logging()
    config_help = "\n\nConfig parameters:\n\n" + "\n".join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument(
        "--rank",
        type=int,
        default=SINGLE_TRAINER,
        help="For multi-machine, this machine's rank",
    )
    opt = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, opt.param)

    enlarge_checkpoint_files(config)

