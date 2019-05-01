#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, NamedTuple, Optional

import torch.distributed as td
import torch.multiprocessing as mp

from torchbiggraph.types import Rank
from torchbiggraph.util import log


class ProcessRanks(NamedTuple):
    """Assign a unique ordinal rank to each process for distributed training.

    torch.distributed requires that N communicating processes register
    themselves with globally unique ranks [0, ..., N-1]. Distributed training
    launches several communicating subprocesses on each machine. This class
    manages the assignment from processes/subprocesses to ranks.
    """

    world_size: int
    trainers: List[Rank]
    parameter_servers: List[Rank]
    parameter_clients: List[Rank]
    lock_server: Rank
    partition_servers: List[Rank]

    @classmethod
    def from_num_invocations(
        cls,
        num_machines: int,
        num_partition_servers: int,
    ) -> 'ProcessRanks':
        world_size = 0

        def add_group(group_size: int) -> List[Rank]:
            nonlocal world_size
            group = [Rank(world_size + r) for r in range(group_size)]
            world_size += group_size
            return group

        trainers = add_group(num_machines)
        parameter_servers = add_group(num_machines)
        parameter_clients = add_group(num_machines)
        lock_server, = add_group(1)
        if num_partition_servers < 0:
            # Use machines as partition servers
            partition_servers = add_group(num_machines)
        else:
            partition_servers = add_group(num_partition_servers)

        return cls(
            world_size, trainers, parameter_servers,
            parameter_clients, lock_server, partition_servers)


def init_process_group(
    init_method: Optional[str],
    world_size: int,
    rank: Rank,
    groups: List[List[Rank]],
    backend: str = "gloo",
) -> List['td.ProcessGroup']:
    # With the THD backend there were no timeouts so high variance in
    # execution time between trainers was not a problem. With the new c10d
    # implementation we do have to take timeouts into account. To simulate
    # the old behavior we use a ridiculously high default timeout.
    timeout = timedelta(days=365)
    log("init_process_group start")
    if init_method is None:
        raise RuntimeError("distributed_init_method must be set when num_machines > 1")
    td.init_process_group(backend,
                          init_method=init_method,
                          world_size=world_size,
                          rank=rank,
                          timeout=timeout)
    log("init_process_group creating groups")
    group_objs = []
    for group in groups:
        group_objs.append(td.new_group(group, timeout=timeout))
    log("init_process_group done")
    return group_objs


class Startable(ABC):

    @abstractmethod
    def start(self) -> None:
        pass


def _server_init(
    server: Startable,
    init_method: Optional[str],
    world_size: int,
    server_rank: Rank,
    groups: List[List[Rank]],
) -> None:
    init_process_group(
        init_method=init_method,
        world_size=world_size,
        rank=server_rank,
        groups=groups,
    )
    server.start()


def start_server(
    server: Startable,
    init_method: Optional[str],
    world_size: int,
    server_rank: Rank,
    groups: List[List[Rank]],
) -> mp.Process:
    p = mp.Process(
        name="%s-%d" % (type(server).__name__, server_rank),
        target=_server_init,
        args=(server, init_method, world_size, server_rank, groups),
    )
    p.daemon = True
    p.start()
    return p
