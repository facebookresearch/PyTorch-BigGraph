#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from torch_extensions.rpc.rpc import Client, Server

from .config import BucketOrder
from .distributed import Startable
from .types import Bucket, EntityName, Partition, Rank, Side
from .util import log, vlog


###
###   Bucket scheduling interface.
###

class AbstractBucketScheduler(ABC):

    @abstractmethod
    def new_pass(self, is_first: bool) -> None:
        pass

    @abstractmethod
    def acquire_bucket(self) -> Tuple[Optional[Bucket], int]:
        pass

    @abstractmethod
    def release_bucket(self, bucket: Bucket) -> None:
        pass

    @abstractmethod
    def check_and_set_dirty(self, entity: EntityName, part: Partition) -> bool:
        pass

    @abstractmethod
    def peek(self) -> Optional[Bucket]:
        pass


###
###   Implementation for single-machine mode.
###

def create_ordered_buckets(
    nparts_lhs: int,
    nparts_rhs: int,
    order: BucketOrder,
    *,
    generator: random.Random,
) -> List[Bucket]:
    if order is BucketOrder.RANDOM:
        return create_buckets_ordered_randomly(
            nparts_lhs, nparts_rhs, generator=generator)
    elif order is BucketOrder.AFFINITY:
        return create_buckets_ordered_by_affinity(
            nparts_lhs, nparts_rhs, generator=generator)
    elif order is BucketOrder.INSIDE_OUT or order is BucketOrder.OUTSIDE_IN:
        return create_buckets_ordered_by_layer(
            nparts_lhs, nparts_rhs, order, generator=generator)
    else:
        raise NotImplementedError("Unknown bucket order: %s" % order)


def create_buckets_ordered_lexicographically(
    nparts_lhs: int,
    nparts_rhs: int,
) -> List[Bucket]:
    """Return buckets in increasing LHS and, for the same LHS, in increasing RHS

    """
    buckets = [
        Bucket(Partition(lhs), Partition(rhs))
        for lhs in range(nparts_lhs)
        for rhs in range(nparts_rhs)
    ]
    return buckets


def create_buckets_ordered_randomly(
    nparts_lhs: int,
    nparts_rhs: int,
    *,
    generator: random.Random,
) -> List[Bucket]:
    """Return all buckets, randomly permuted.

    Produce buckets for [0, #LHS) x [0, #RHS) and shuffle them.

    """
    buckets = create_buckets_ordered_lexicographically(nparts_lhs, nparts_rhs)
    generator.shuffle(buckets)
    return buckets


def create_buckets_ordered_by_affinity(
    nparts_lhs: int,
    nparts_rhs: int,
    *,
    generator: random.Random,
) -> List[Bucket]:
    """Try having consecutive buckets share as many partitions as possible.

    Start from a random bucket. Until there are buckets left, try to choose the
    next one so that it has as many partitions in common as possible with the
    previous one. When multiple options are available, pick one randomly.

    """
    if nparts_lhs <= 0 or nparts_rhs <= 0:
        return []

    # This is our "source of truth" on what buckets we haven't outputted yet. It
    # can be queried in constant time.
    remaining: Set[Bucket] = set()
    # These are our random orders: we shuffle them once and then pop from the
    # end. Each bucket appears in several of them. They are updated lazily,
    # which means they may contain buckets that have already been outputted.
    all_buckets: List[Bucket] = []
    buckets_per_partition: List[List[Bucket]] = \
        [[] for _ in range(max(nparts_lhs, nparts_rhs))]

    for lhs in range(nparts_lhs):
        for rhs in range(nparts_rhs):
            b = Bucket(Partition(lhs), Partition(rhs))
            remaining.add(b)
            all_buckets.append(b)
            buckets_per_partition[lhs].append(b)
            buckets_per_partition[rhs].append(b)

    generator.shuffle(all_buckets)
    for buckets in buckets_per_partition:
        generator.shuffle(buckets)

    b = all_buckets.pop()
    remaining.remove(b)
    order = [b]

    while remaining:
        transposed_b = Bucket(b.rhs, b.lhs)
        if transposed_b in remaining:
            remaining.remove(transposed_b)
            order.append(transposed_b)
            if not remaining:
                break

        same_as_lhs = buckets_per_partition[b.lhs]
        same_as_rhs = buckets_per_partition[b.rhs]
        while len(same_as_lhs) > 0 or len(same_as_rhs) > 0:
            chosen, = generator.choices(
                [same_as_lhs, same_as_rhs],
                weights=[len(same_as_lhs), len(same_as_rhs)],
            )
            next_b = chosen.pop()
            if next_b in remaining:
                break
        else:
            while True:
                next_b = all_buckets.pop()
                if next_b in remaining:
                    break
        remaining.remove(next_b)
        order.append(next_b)
        b = next_b

    return order


def create_layer_of_buckets(
    nparts_lhs: int,
    nparts_rhs: int,
    layer_idx: int,
    *,
    generator: random.Random,
) -> List[Bucket]:
    """Return the layer of #LHS x #RHS matrix of the given index

    The i-th layer contains the buckets (lhs, rhs) such that min(lhs, rhs) == i.
    Buckets that are one the transpose of the other will be consecutive. Other
    than that, the order is random.

    """
    layer_p = Partition(layer_idx)
    pairs = [[Bucket(layer_p, layer_p)]]
    for idx in range(layer_idx + 1, max(nparts_lhs, nparts_rhs)):
        p = Partition(idx)
        pair = []
        if p < nparts_lhs:
            pair.append(Bucket(p, layer_p))
        if p < nparts_rhs:
            pair.append(Bucket(layer_p, p))
        generator.shuffle(pair)
        pairs.append(pair)
    generator.shuffle(pairs)
    return [b for p in pairs for b in p]


def create_buckets_ordered_by_layer(
    nparts_lhs: int,
    nparts_rhs: int,
    order: BucketOrder,
    *,
    generator: random.Random,
) -> List[Bucket]:
    """Output buckets in concentric L-shaped layers (e.g., first row + column)

    If order is OUTSIDE_IN, start outputting all buckets that have 0 as one of
    their partitions. Once done, output all the ones (among those remaining)
    that have 1 as one of their partitions. After that, those that have 2, and
    so on, until none are left. Each of these stages is called a layer, and
    within each layer buckets are shuffled at random.

    For example: [
        (2, 0), (0, 3), (0, 1), (1, 0), (0, 2), (0, 0),  # Layer for 0
        (1, 2), (1, 1), (2, 1), (1, 3),  # Layer for 1
        (2, 2), (2, 3),  # Layer for 2
    ]

    If order is INSIDE_OUT, the layers are the same but their order is reversed.

    When displaying the layers on a #LHS x #RHS matrix, they have an upside-down
    "L" shape, with the layer for 0 being comprised of the first row and column,
    and the subsequent ones being "nested" inside of it. Graphically:

       |  0   1   2   3
    ---+----------------
     0 | L0  L0  L0  L0
     1 | L0  L1  L1  L1
     2 | L0  L1  L2  L2

    """
    if order is not BucketOrder.INSIDE_OUT \
            and order is not BucketOrder.OUTSIDE_IN:
        raise ValueError("Unknown order: %s" % order)

    layers = [
        create_layer_of_buckets(nparts_lhs, nparts_rhs, i, generator=generator)
        for i in range(min(nparts_lhs, nparts_rhs))
    ]
    if order is BucketOrder.INSIDE_OUT:
        layers.reverse()
    return [b for l in layers for b in l]


class SingleMachineBucketScheduler(AbstractBucketScheduler):

    def __init__(self, nparts_lhs: int, nparts_rhs: int, order: BucketOrder) -> None:
        self.nparts_lhs = nparts_lhs
        self.nparts_rhs = nparts_rhs
        self.order = order

        self.buckets: List[Bucket] = []

    def new_pass(self, is_first: bool) -> None:
        self.buckets = create_ordered_buckets(
            nparts_lhs=self.nparts_lhs,
            nparts_rhs=self.nparts_rhs,
            order=self.order,
            generator=random.Random(),
        )

        # Print buckets
        vlog('\nPartition pairs:')
        for bucket in self.buckets:
            vlog("%s" % (bucket,))
        vlog('')

    def acquire_bucket(self) -> Tuple[Optional[Bucket], int]:
        try:
            bucket = self.buckets.pop(0)
        except IndexError:
            return None, 0
        remaining = len(self.buckets)
        return bucket, remaining

    def release_bucket(self, bucket: Bucket) -> None:
        pass

    def check_and_set_dirty(self, entity: EntityName, part: Partition) -> bool:
        return False

    def peek(self) -> Optional[Bucket]:
        try:
            return self.buckets[0]
        except IndexError:
            return None


###
###   Implementation for distributed training mode.
###

class LockServer(Server, Startable):

    def __init__(
        self,
        num_clients: int,
        nparts_lhs: int,
        nparts_rhs: int,
        lock_lhs: bool,
        lock_rhs: bool,
        init_tree: bool,
    ) -> None:
        super().__init__(num_clients)
        self.buckets: List[Bucket] = \
            create_buckets_ordered_lexicographically(nparts_lhs, nparts_rhs)
        self.lock_lhs: bool = lock_lhs
        self.lock_rhs: bool = lock_rhs
        self.locked_sides: List[Side] = []
        if lock_lhs:
            self.locked_sides.append(Side.LHS)
        if lock_rhs:
            self.locked_sides.append(Side.RHS)
        self.init_tree = init_tree

        self.active: Dict[Bucket, Rank] = {}
        self.done: Set[Bucket] = set()
        self.dirty: Set[Tuple[EntityName, Partition]] = set()
        self.initialized_partitions: Optional[Set[Partition]] = None

    def new_pass(self, is_first: bool = False) -> None:
        """Start a new epoch of training."""
        self.active = {}
        self.done = set()
        self.dirty = set()
        if self.init_tree and is_first:
            self.initialized_partitions = {Partition(0)}
        else:
            self.initialized_partitions = None

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

    def acquire_bucket(
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
                and (self.initialized_partitions is None
                     or pair.lhs in self.initialized_partitions
                     or pair.rhs in self.initialized_partitions)):

                self.active[pair] = rank
                self.done.add(pair)
                if self.initialized_partitions is not None:
                    self.initialized_partitions.add(pair.lhs)
                    self.initialized_partitions.add(pair.rhs)
                log("lockserver %d acquire %s: active= %s" % (rank, pair, self.active))
                return pair, remaining

        return None, remaining

    def release_bucket(self, bucket: Bucket) -> None:
        """
        Releases the lock on lhs and rhs, and marks this pair as done.
        """
        if bucket.lhs is not None:
            self.active.pop(bucket)
            log("lockserver release %s: active= %s" % (bucket, self.active))

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

    def __init__(self, server_rank: Rank) -> None:
        super().__init__(LockServer, server_rank)


class DistributedBucketScheduler(AbstractBucketScheduler):

    def __init__(self, server_rank: Rank, client_rank: Rank):
        self.client = LockClient(server_rank)
        self.rank = client_rank

        self.old_b: Optional[Bucket] = None

    def new_pass(self, is_first: bool) -> None:
        if self.rank == 0:
            self.client.new_pass(is_first)
        self.old_b = None

    def acquire_bucket(self) -> Tuple[Optional[Bucket], int]:
        bucket, remaining = self.client.acquire_bucket(self.rank, maybe_old_bucket=self.old_b)
        if bucket is not None:
            self.old_b = bucket
        return bucket, remaining

    def release_bucket(self, bucket: Bucket) -> None:
        self.client.release_bucket(bucket)

    def check_and_set_dirty(self, entity: EntityName, part: Partition) -> bool:
        return self.client.check_and_set_dirty(entity, part)

    def peek(self) -> Optional[Bucket]:
        return None
