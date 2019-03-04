#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import Dict, List, Optional, Set, Tuple

import torch.multiprocessing as mp
from torch_extensions.rpc.rpc import Client, Server

from .util import log, init_process_group, Side, Bucket, Partition, EntityName, Rank


class LockServer(Server):

    def new_epoch(
        self,
        buckets: List[Bucket],
        lock_lhs: bool = True,
        lock_rhs: bool = True,
        init_tree: bool = False,
    ) -> None:
        """
        Start a new epoch of training.

        Args:
            buckets: A list of 2-tuples of all partition pairs that should be
                   processed during this epoch.
        """
        self.active: Dict[Bucket, Rank] = {}
        self.done: Set[Bucket] = set()
        self.dirty: Set[Tuple[EntityName, Partition]] = set()
        self.buckets: List[Bucket] = buckets
        self.locked_sides: List[Side] = []
        if lock_lhs:
            self.locked_sides.append(Side.LHS)
        if lock_rhs:
            self.locked_sides.append(Side.RHS)

        self.init_tree: Optional[Set[Partition]] = {Partition(0)} if init_tree else None

    def _can_acquire(
        self,
        rank: Rank,
        part: Partition,
        locked_parts: Dict[Partition, Rank],
        side: Side,
    ) -> bool:
        if side not in self.locked_sides:
            return True
        return part not in locked_parts or locked_parts[part] == rank

    def acquire_pair(
        self,
        rank: Rank,
        maybe_old_bucket: Optional[Bucket] = None,
    ) -> Tuple[Optional[Bucket], int]:
        """
        Finds a (lhs, rhs) partition pair that has not already been acquired
        this epoch, and where neither the lhs nor rhs partitions are currently
        locked. Locks this lhs and rhs until `release_pair` is called. Will try
        to find a pair that has the same lhs (if not, rhs) as old_bucket.

        If no pair is available, returns None.

        Returns:
            pair: a (lhs, rhs) partition pair. lhs and rhs are locked until
                  `release_pair` is called.
                  If no pair is available, None is returned.
            remaining: The number of pairs remaining. When this is 0 then the
                       epoch is done.
        """
        remaining = len(self.buckets) - len(self.done)
        if maybe_old_bucket is not None:
            # The linter isn't smart enough to figure out that the closure is
            # capturing a non-None value, thus alias it to a new variable, which
            # will get a non-Optional type.
            old_bucket = maybe_old_bucket  # The linter isn't too smart around closures...
            ordered_buckets = sorted(
                self.buckets, key=lambda x: - (2 * (x.lhs == old_bucket.lhs)
                                               + (x.rhs == old_bucket.rhs)))
        else:
            ordered_buckets = self.buckets

        locked_partitions = {
            bucket.get_partition(side): rank
            for bucket, rank in self.active.items()
            for side in self.locked_sides
        }

        for pair in ordered_buckets:
            if (pair not in self.done
                and self._can_acquire(rank, pair.lhs, locked_partitions, Side.LHS)
                and self._can_acquire(rank, pair.rhs, locked_partitions, Side.RHS)
                and (self.init_tree is None
                     or pair.lhs in self.init_tree
                     or pair.rhs in self.init_tree)):

                self.active[pair] = rank
                self.done.add(pair)
                if self.init_tree is not None:
                    self.init_tree.add(pair.lhs)
                    self.init_tree.add(pair.rhs)
                print("lockserver %d acquire %s: active= %s" %
                      (rank, pair, self.active), file=sys.stderr)
                sys.stderr.flush()
                return pair, remaining

        return None, remaining

    def release_bucket(self, bucket: Bucket) -> None:
        """
        Releases the lock on lhs and rhs, and marks this pair as done.
        """
        if bucket.lhs is not None:
            self.active.pop(bucket)
            print("lockserver release %s: active= %s" %
                  (bucket, self.active), file=sys.stderr)
            sys.stderr.flush()

    def check_and_set_dirty(self, entity: EntityName, part: Partition) -> bool:
        """
        Keeps track over an epoch of which (entity, part) pairs have been
        processed. Since we store partition data in temporary files during an
        epoch, this dirty state is necessary to know whether the partition
        checkpint or the intermediate file should be read.
        """
        key = (entity, part)
        res = key in self.dirty
        self.dirty.add(key)
        return res


class LockClient(Client):

    def __init__(self, server_rank):
        super(LockClient, self).__init__(LockServer, server_rank)


def _start_lock_server(init_method, world_size, num_clients, server_rank, groups):
    log("_start_lock_server: init_process_group begin")
    init_process_group(init_method=init_method,
                       world_size=world_size,
                       rank=server_rank,
                       groups=groups)
    s = LockServer(num_clients)
    s.start()


def setup_lock_server(is_server_node, server_rank, world_size, num_clients, init_method, groups):

    if is_server_node:
        # set up the parameter server on rank 0, but as a separate node
        # with MPI rank provided
        p_server = mp.Process(target=_start_lock_server,
                              args=(init_method, world_size, num_clients, server_rank, groups)
                              )
        p_server.daemon = True
        p_server.start()

    client = LockClient(server_rank)

    return client
