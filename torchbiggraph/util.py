#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import os
import os.path
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
from torch.optim import Optimizer

from torchbiggraph.config import ConfigSchema
from torchbiggraph.types import EntityName, FloatTensorType, Side


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


def round_up_to_nearest_multiple(value: int, factor: int) -> int:
    return ((value - 1) // factor + 1) * factor


def fast_approx_rand(numel: int) -> FloatTensorType:
    if numel < 1_000_003:
        tensor = torch.randn(numel)
        # share_memory_ does return the tensor but its type annotation says it
        # doesn't, thus we do this in two separate steps.
        tensor.share_memory_()
        return tensor
    # construct the tensor storage in shared mem so we don't have to copy it
    storage = torch.FloatStorage._new_shared(numel)
    tensor = torch.FloatTensor(storage)
    rand = torch.randn(1_000_003)
    excess = numel % 1_000_003
    # Using just `-excess` would give bad results when excess == 0.
    tensor[:numel - excess].view(-1, 1_000_003)[...] = rand
    tensor[numel - excess:] = rand[:excess]
    return tensor


class DummyOptimizer(Optimizer):

    def __init__(self) -> None:
        # This weird dance makes Optimizer accept an empty parameter list.
        super().__init__([{'params': []}], {})

    def step(self, closure: None = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def share_memory(self) -> None:
        pass


# HOGWILD

def _pool_init():
    torch.set_num_threads(1)
    torch.manual_seed(os.getpid())


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
    return mp.get_context("spawn").Pool(num_workers, initializer=_pool_init)


# config routines

def get_partitioned_types(
    config: ConfigSchema,
    side: Side,
) -> Tuple[int, Set[EntityName]]:
    """Return the number of partitions on a given side and the partitioned entity types

    Each of the entity types that appear on the given side (LHS or RHS) of a relation
    type is split into some number of partitions. The ones that are split into one
    partition are called "unpartitioned" and behave as if all of their entities
    belonged to all buckets. The other ones are the "properly" partitioned ones.
    Currently, they must all be partitioned into the same number of partitions. This
    function returns that number and the names of the properly partitioned entity
    types.

    """
    entity_names_by_num_parts: Dict[int, Set[EntityName]] = defaultdict(set)
    for relation_config in config.relations:
        entity_name = side.pick(relation_config.lhs, relation_config.rhs)
        entity_config = config.entities[entity_name]
        entity_names_by_num_parts[entity_config.num_partitions].add(entity_name)

    if 1 in entity_names_by_num_parts:
        del entity_names_by_num_parts[1]

    if len(entity_names_by_num_parts) == 0:
        return 1, set()
    if len(entity_names_by_num_parts) > 1:
        raise RuntimeError("Currently num_partitions must be a single "
                           "value across all partitioned entities.")

    (num_partitions, partitioned_entity_names), = entity_names_by_num_parts.items()
    return num_partitions, partitioned_entity_names


# compute a randomized AUC using a fixed number of sample points
# NOTE: AUC is the probability that a randomly chosen positive example
# has a higher score than a randomly chosen negative example
def compute_randomized_auc(
    pos_: FloatTensorType,
    neg_: FloatTensorType,
    num_samples: int,
) -> float:
    pos_, neg_ = pos_.view(-1), neg_.view(-1)
    diff = (pos_[torch.randint(len(pos_), (num_samples,))]
            > neg_[torch.randint(len(neg_), (num_samples,))])
    return float(diff.float().mean())


def get_num_workers(override: Optional[int]) -> int:
    if override is not None:
        return override
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count
    result = 40
    print("WARNING: number of workers unspecified and couldn't autodetect "
          "CPU count; defaulting to %d workers." % result)
    return result
