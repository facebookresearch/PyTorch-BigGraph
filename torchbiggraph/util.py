#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, List, NamedTuple, NewType, Optional, \
    Set, Tuple, TypeVar

import torch
import torch.multiprocessing as mp
from torch.optim import Optimizer

from .config import BucketOrder, ConfigSchema
from .entitylist import EntityList


T = TypeVar("T")


class Side(Enum):
    LHS = 0
    RHS = 1

    def pick(self, lhs: T, rhs: T) -> T:
        if self is Side.LHS:
            return lhs
        elif self is Side.RHS:
            return rhs
        else:
            raise NotImplementedError("Unknown side: %s" % self)


EntityName = NewType("EntityName", str)
Rank = NewType("Rank", int)
Partition = NewType("Partition", int)
ModuleStateDict = NewType("ModuleStateDict", Dict[str, torch.Tensor])
OptimizerStateDict = NewType("OptimizerStateDict", Dict[str, Any])


class Bucket(NamedTuple):
    lhs: Partition
    rhs: Partition

    def get_partition(self, side: Side) -> Partition:
        return side.pick(self.lhs, self.rhs)

    def __str__(self):
        return "( %d , %d )" % (self.lhs, self.rhs)


def log(msg: str) -> None:
    """Log msg to stdout with a timestamp. Flush stdout.
    """
    print("%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


_verbosity_level = 0


def vlog(msg: str, level: int = 1) -> None:
    if _verbosity_level >= level:
        log(msg)


def split_almost_equally(size: int, *, num_parts: int) -> Iterable[slice]:
    """Split an interval of the given size into the given number of subintervals

    The sizes of the subintervals will be between the floor and the ceil of the
    "exact" fractional size, with larger intervals preceding smaller ones.

    """
    size_per_part = size // num_parts
    num_larger_parts = size % num_parts
    prev = 0
    for i in range(num_parts):
        next_ = prev + size_per_part + (1 if i < num_larger_parts else 0)
        yield slice(prev, next_)
        prev = next_


def chunk_by_index(index: torch.Tensor, *others: EntityList) -> List[List[EntityList]]:
    """
    Parameters:
        index: An integral-valued 1D tensor of length N containing indexes of each row.
        *others: Other tensors with first dimension of length N, indexed by index.
    Returns:
        A list of length N, where each element i is a tuple of tensors containing
        the subset of rows where index[rows] == i.
    """
    def slice(x, begin, end):
        if begin == end:
            return x.new()
        else:
            return x[begin:end]

    sorted_index, order = index.sort()
    nchunks = sorted_index[-1] + 1
    N = sorted_index.size(0)
    delta = sorted_index[1:] - sorted_index[:-1]
    cutpoints = delta.nonzero()
    if cutpoints.nelement() > 0:
        cutpoints = cutpoints[:, 0]
        jumps = delta[cutpoints]
        cutpoints += 1

        cutpoints = torch.cat([
            cutpoints.new([0]),
            cutpoints,
            cutpoints.new([N])])
        jumps = torch.cat([
            jumps.new([sorted_index[0]]),
            jumps,
            jumps.new([nchunks - sorted_index[-1]])])
    else:
        cutpoints = cutpoints.new([0, N])
        jumps = delta.new([sorted_index[0], nchunks - sorted_index[-1]])

    sorted_tensors = [sorted_index] + [T[order] for T in others]

    chunked: List[List[EntityList]] = [[] for _ in sorted_tensors]
    begin_row = 0
    for cur_row, jump in zip(cutpoints, jumps):
        for _ in range(jump):
            for c, s in zip(chunked, sorted_tensors):
                c.append(slice(s, begin_row, cur_row))
            begin_row = cur_row

    return chunked


def fast_approx_rand(numel: int) -> torch.FloatTensor:
    if numel < 1_000_003:
        return torch.randn(numel).share_memory_()
    # construct the tensor storage in shared mem so we don't have to copy it
    storage = torch.FloatStorage._new_shared(numel)
    tensor = torch.FloatTensor(storage)
    rand = torch.randn(1_000_003)
    excess = numel % 1_000_003
    # Using just `-excess` would give bad results when excess == 0.
    tensor[:numel - excess].view(-1, 1_000_003)[...] = rand
    tensor[numel - excess:] = rand[:excess]
    return tensor


def infer_input_index_base(config: ConfigSchema) -> int:
    """Infer whether input data has indices starting at 0 or at 1.

    Torchbiggraph used to use 1-based indexing. It now supports (and prefers)
    0-based indexing as well. To keep backwards compatibility, it auto-detects
    the format of the input data and sticks to the same format in the output.

    """
    one_based: Set[bool] = set()
    one_based.update(
        os.path.exists(os.path.join(path, "edges_1_1.h5"))
        and not os.path.exists(os.path.join(path, "edges_0_0.h5"))
        for path in config.edge_paths
    )
    one_based.update(
        os.path.exists(os.path.join(config.entity_path, "entity_count_%s_1.pt" % entity))
        and not os.path.exists(os.path.join(config.entity_path, "entity_count_%s_0.pt" % entity))
        for entity in config.entities
    )
    if len(one_based) != 1:
        raise RuntimeError(
            "Cannot determine whether the input files are using 0- or 1-based "
            "indexing. Either they are in a mixed format, or some files are "
            "missing or some ther I/O error occurred."
        )
    return 1 if one_based.pop() else 0


class DummyOptimizer(Optimizer):

    def __init__(self):
        # This weird dance makes Optimizer accept an empty parameter list.
        super().__init__([{'params': []}], {})

    def step(self, closure=None):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    def share_memory(self):
        pass


# HOGWILD

def create_pool(num_workers: int) -> mp.Pool:
    # PyTorch relies on OpenMP, which by default parallelizes operations by
    # implicitly spawning as many threads as there are cores, and synchronizing
    # them with each other. This interacts poorly with Hogwild!-style subprocess
    # pools as if each child process spawns its own OpenMP threads there can
    # easily be thousands of threads that mostly wait in barriers. Calling
    # set_num_threads(1) in both the parent and children prevents this.
    # OpenMP can also lead to deadlocks if it gets initialized in the parent
    # process before the fork (this bit us in unit tests, due to the generation
    # of the test input data). Using the "spawn" context (i.e., fork + exec)
    # solved the issue in most cases but still left some deadlocks. See
    # https://github.com/pytorch/pytorch/issues/17199 for some more information
    # and discussion.
    torch.set_num_threads(1)
    return mp.Pool(num_workers, initializer=torch.set_num_threads, initargs=(1,))


# config routines

def get_partitioned_types(
    config: ConfigSchema,
) -> Tuple[int, int, Set[EntityName], Set[EntityName]]:
    # Currently, we only allow a single value of num_partitions other than 1.
    # Eventually, we will allow arbitrary nested num_partitions.
    max_parts = max(e.num_partitions for e in config.entities.values())
    for e in config.entities.values():
        assert e.num_partitions == 1 or e.num_partitions == max_parts, (
            "Currently num_partitions must be either 1 or a single value across "
            "all entities.")

    # Figure out how many lhs and rhs partitions we need
    nparts_rhs, nparts_lhs = 1, 1
    lhs_partitioned_types, rhs_partitioned_types = set(), set()
    for relation in config.relations:
        if config.entities[relation.lhs].num_partitions != 1:
            lhs_partitioned_types.add(EntityName(relation.lhs))
            nparts_lhs = max_parts
        if config.entities[relation.rhs].num_partitions != 1:
            rhs_partitioned_types.add(EntityName(relation.rhs))
            nparts_rhs = max_parts

    return nparts_lhs, nparts_rhs, lhs_partitioned_types, rhs_partitioned_types


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


# compute a randomized AUC using a fixed number of sample points
# NOTE: AUC is the probability that a randomly chosen positive example
# has a higher score than a randomly chosen negative example
def compute_randomized_auc(
    pos_: torch.FloatTensor,
    neg_: torch.FloatTensor,
    num_samples: int,
) -> float:
    pos_, neg_ = pos_.view(-1), neg_.view(-1)
    diff = pos_[torch.randint(len(pos_), (num_samples,))] \
           > neg_[torch.randint(len(neg_), (num_samples,))]
    return float(diff.float().mean())


# makes it easy to pass around process group args as a dict and pass as kwargs
def init_process_group(
    init_method: Optional[str],
    world_size: int,
    rank: Rank,
    groups: List[List[int]],
    backend: str = "gloo",
) -> List['torch.distributed.ProcessGroup']:
    # With the THD backend there were no timeouts so high variance in
    # execution time between trainers was not a problem. With the new c10d
    # implementation we do have to take timeouts into account. To simulate
    # the old behavior we use a ridiculously high default timeout.
    timeout = timedelta(days=365)
    log("init_process_group start")
    if init_method is None:
        raise RuntimeError("distributed_init_method must be set when num_machines > 1")
    torch.distributed.init_process_group(backend,
                                         init_method=init_method,
                                         world_size=world_size,
                                         rank=rank,
                                         timeout=timeout)
    log("init_process_group creating groups")
    group_objs = []
    for group in groups:
        group_objs.append(torch.distributed.new_group(group, timeout=timeout))
    log("init_process_group done")
    return group_objs
