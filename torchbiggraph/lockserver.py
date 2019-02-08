#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch.multiprocessing as mp
from torch_extensions.rpc.rpc import Client, Server

from .util import log, init_process_group, Side


class Bucket(NamedTuple):
    lhs: int
    rhs: int

    def __str__(self):
        return "( %d , %d )" % (self.lhs, self.rhs)


class LockServer(Server):

    def new_epoch(
        self,
        pairs: List[Bucket],
        lock_lhs=True,
        lock_rhs=True,
        init_tree=False
    ):
        """
        Start a new epoch of training.

        Args:
            pairs: A list of 2-tuples of all partition pairs that should be
                   processed during this epoch.
        """
        self.active: Dict[Bucket, int] = {}
        self.done: Set[Bucket] = set()
        self.dirty: Set[Tuple[int, int]] = set()
        self.pairs: List[Bucket] = pairs
        self.do_lock: Tuple[bool, bool] = (lock_lhs, lock_rhs)
        self.init_tree: Optional[Set[int]] = {0} if init_tree else None

    def _can_acquire(
        self,
        rank: int,
        part: int,
        locked_parts: Dict[int, int],
        side: Side,
    ) -> bool:
        if not side.pick_tuple(self.do_lock):
            return True
        return part not in locked_parts or locked_parts[part] == rank

    def acquire_pair(
        self,
        rank: int,
        maybe_oldP: Optional[Bucket]=None,
    ) -> Tuple[Optional[Bucket], int]:
        """
        Finds a (lhs, rhs) partition pair that has not already been acquired
        this epoch, and where neither the lhs nor rhs partitions are currently
        locked. Locks this lhs and rhs until `release_pair` is called. Will try
        to find a pair that has the same lhs (if not, rhs) as oldP.

        If no pair is available, returns None.

        Returns:
            pair: a (lhs, rhs) partition pair. lhs and rhs are locked until
                  `release_pair` is called.
                  If no pair is available, None is returned.
            remaining: The number of pairs remaining. When this is 0 then the
                       epoch is done.
        """
        remaining = len(self.pairs) - len(self.done)
        if maybe_oldP is not None:
            oldP = maybe_oldP  # The linter isn't too smart around closures...
            ordered_pairs = sorted(self.pairs, key=lambda x:
                                   -((x.lhs == oldP.lhs) * 2 +
                                     (x.rhs == oldP.rhs)))
        else:
            ordered_pairs = self.pairs

        locked_parts = {p[i]: rank for p, rank in self.active.items()
                        for i in range(2) if self.do_lock[i]}

        for pair in ordered_pairs:
            if (pair not in self.done
                and self._can_acquire(rank, pair.lhs, locked_parts, Side.LHS)
                and self._can_acquire(rank, pair.rhs, locked_parts, Side.RHS)
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

    def release_pair(self, pair: Bucket):
        """
        Releases the lock on lhs and rhs, and marks this pair as done.
        """
        if pair.lhs is not None:
            self.active.pop(pair)
            print("lockserver release %s: active= %s" %
                  (pair, self.active), file=sys.stderr)
            sys.stderr.flush()

    def check_and_set_dirty(self, entity: int, part: int) -> bool:
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
