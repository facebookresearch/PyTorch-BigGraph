#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from itertools import chain

import torch.distributed as td

from torchbiggraph.config import ConfigSchema, parse_config
from torchbiggraph.distributed import ProcessRanks, init_process_group
from torchbiggraph.parameter_sharing import ParameterServer

# This is a small binary that just runs a partition server.
# You need to run this if you run a distributed run and set
# num_partition_servers > 1.


def run_partition_server(config, rank=0):
    if config.num_partition_servers <= 0:
        raise RuntimeError("Config doesn't require explicit partition servers")
    if not 0 <= rank < config.num_partition_servers:
        raise RuntimeError("Invalid rank for partition server")
    if not td.is_available():
        raise RuntimeError("The installed PyTorch version doesn't provide "
                           "distributed training capabilities.")
    ranks = ProcessRanks.from_num_invocations(
        config.num_machines, config.num_partition_servers)
    init_process_group(
        rank=ranks.partition_servers[rank],
        world_size=ranks.world_size,
        init_method=config.distributed_init_method,
        groups=[ranks.trainers],
    )
    ps = ParameterServer(num_clients=len(ranks.trainers))
    ps.start()


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--rank', type=int, default=0,
                        help="For multi-machine, this machine's rank")
    opt = parser.parse_args()

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
    else:
        overrides = None
    config = parse_config(opt.config, overrides)

    run_partition_server(config, opt.rank)


if __name__ == '__main__':
    main()
