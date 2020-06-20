#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from torchbiggraph.batching import AbstractBatchProcessor
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.train_cpu import TrainingCoordinator
from torchbiggraph.train_gpu import GPUTrainingCoordinator
from torchbiggraph.types import SINGLE_TRAINER, Rank
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
)


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


def train(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    trainer: Optional[AbstractBatchProcessor] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    rank: Rank = SINGLE_TRAINER,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> None:
    CoordinatorT = (
        GPUTrainingCoordinator if config.num_gpus > 0 else TrainingCoordinator
    )
    coordinator = CoordinatorT(config, model, trainer, evaluator, rank, subprocess_init)
    coordinator.train()
    coordinator.close()


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
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    train(config, rank=opt.rank, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
