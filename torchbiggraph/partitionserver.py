#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
from typing import Callable, List, Optional

import torch.distributed as td
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.distributed import ProcessRanks, init_process_group
from torchbiggraph.parameter_sharing import ParameterServer
from torchbiggraph.types import SINGLE_TRAINER, Rank
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")


# This is a small binary that just runs a partition server.
# You need to run this if you run a distributed run and set
# num_partition_servers > 1.


def run_partition_server(
    config: ConfigSchema,
    rank: Rank = SINGLE_TRAINER,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> None:
    tag_logs_with_process_name(f"PartS-{rank}")
    if config.num_partition_servers <= 0:
        raise RuntimeError("Config doesn't require explicit partition servers")
    if not 0 <= rank < config.num_partition_servers:
        raise RuntimeError("Invalid rank for partition server")
    if not td.is_available():
        raise RuntimeError(
            "The installed PyTorch version doesn't provide "
            "distributed training capabilities."
        )
    ranks = ProcessRanks.from_num_invocations(
        config.num_machines, config.num_partition_servers
    )

    num_ps_groups = config.num_groups_for_partition_server
    groups: List[List[int]] = [ranks.trainers]  # barrier group
    groups += [ranks.trainers + ranks.partition_servers] * num_ps_groups  # ps groups
    group_idxs_for_partition_servers = range(1, len(groups))

    if subprocess_init is not None:
        subprocess_init()
    groups = init_process_group(
        rank=ranks.partition_servers[rank],
        world_size=ranks.world_size,
        init_method=config.distributed_init_method,
        groups=groups,
    )
    ps = ParameterServer(
        num_clients=len(ranks.trainers),
        group_idxs=group_idxs_for_partition_servers,
        log_stats=True,
    )
    ps.start(groups)
    logger.info("ps.start done")


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

    run_partition_server(config, opt.rank, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
