#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from itertools import chain

from .config import parse_config, ConfigSchema
from .parameterserver import ParameterServer
from .util import init_process_group

# This is a small binary that just runs a partition server.
# You need to run this if you run a distributed run and set
# num_partition_servers > 1.


def run_partition_server(config, rank=0):
    barrier_group_ranks = list(range(config.num_machines))
    init_process_group(rank=config.num_machines * 3 + 1 + rank,
                       init_method=config.distributed_init_method,
                       world_size=config.num_machines * 3 + 1 + config.num_partition_servers,
                       groups=[barrier_group_ranks],  # ugh
                       )
    ps = ParameterServer(config.num_machines)
    ps.start()


def main():
    # torch.multiprocessing.set_start_method("spawn")
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
