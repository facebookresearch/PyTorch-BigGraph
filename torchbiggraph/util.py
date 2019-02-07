#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import queue
import random
import traceback
from datetime import datetime, timedelta
from enum import Enum
from itertools import zip_longest
from typing import List, Tuple, TypeVar

import attr
import numpy as np
import torch
import torch.multiprocessing as mp

from .config import BucketOrder


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

    def pick_tuple(self, pair: Tuple[T, T]) -> T:
        return self.pick(pair[0], pair[1])


def log(msg):
    """Log msg to stdout with a timestamp. Flush stdout.
    """
    print("%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


_verbosity_level = 0


def vlog(msg, level=1):
    if _verbosity_level >= level:
        log(msg)


def chunk_by_index(index, *others):
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

    chunked = [[] for _ in sorted_tensors]
    begin_row = 0
    for cur_row, jump in zip(cutpoints, jumps):
        for _ in range(jump):
            for c, s in zip(chunked, sorted_tensors):
                c.append(slice(s, begin_row, cur_row))
            begin_row = cur_row

    return chunked


def product(L):
    p = 1
    for i in L:
        p *= i
    return p


def fast_approx_rand(*size):
    numel = product(size)
    if numel < 1000003:
        return torch.rand(size)
    # construct the tensor storage in shared mem so we don't have to copy it
    storage = torch.Storage._new_shared(numel)
    res = torch.Tensor(storage).view(*size)
    rand = torch.rand(1000003)
    i = 0
    while i < res.nelement():
        k = min(rand.nelement(), res.nelement() - i)
        res.view(-1)[i:i + k].copy_(rand[:k])
        i += k
    return res


class DummyOptimizer(object):
    def step(self):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass


# HOGWILD

def _loop(rank, qIn, qOut, F, args):
    """Top-level handler for HOGWILD threads. Just calls the main training
    routine.
    """
    while True:
        try:
            data = qIn.get(timeout=0.1)
            if data is None:
                break

            res = F(rank, args, *data)
            del data   # allow memory to be freed
            qOut.put(res)
        except queue.Empty:
            pass
        except BaseException as e:
            traceback.print_exc()
            qOut.put(e)
            raise


def create_workers(N, F, args):
    # setup workers
    torch.set_num_threads(1)
    qIn, qOut = [], []
    processes = []
    for rank in range(N):
        qi, qo = mp.Queue(), mp.Queue()
        p = mp.Process(target=_loop, args=(
            rank, qi, qo, F, args))
        p.daemon = True
        p.start()
        processes.append(p)
        qIn.append(qi)
        qOut.append(qo)

    return processes, qIn, qOut


def join_workers(processes, qIn, qOut):
    # log("Joining workers...")
    for p, q in zip(processes, qIn):
        q.put(None)
        p.join()
    # log("Done joining workers.")


# config routines

def get_partitioned_types(config):
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
            lhs_partitioned_types.add(relation.lhs)
            nparts_lhs = max_parts
        if config.entities[relation.rhs].num_partitions != 1:
            rhs_partitioned_types.add(relation.rhs)
            nparts_rhs = max_parts

    return nparts_lhs, nparts_rhs, lhs_partitioned_types, rhs_partitioned_types


def update_config_for_dynamic_relations(config):
    dynamic_rel_path = os.path.join(config.entity_path, "dynamic_rel_count.pt")
    if os.path.exists(dynamic_rel_path):
        num_dynamic_rels = torch.load(dynamic_rel_path)
        log("Found file at %s ; enabling dynamic rel batches with %d relations" %
            (dynamic_rel_path, num_dynamic_rels))
        assert len(config.relations) == 1, """
            Dynamic relations are enabled so there should only be a single entry
            in config.relations with config for all relations."""
        return attr.evolve(config, num_dynamic_rels=num_dynamic_rels)
    return config


# train
def create_partition_pairs(nparts_lhs, nparts_rhs, bucket_order: BucketOrder):
    """Create all pairs of tuples (<LHS partition ID>, <RHS partition ID>), and
    sort the tuples according to bucket_order.
    """

    if bucket_order is BucketOrder.CHAINED_SYMMETRIC_PAIRS:
        """
        Create an ordering of the partition pairs that ensures that the
        mirror-image of any partition pair is adjacent to it if it exists. For
        instance, if the LHS and RHS each contain 3 partitions, (1, 2) will be
        adjacent to (2, 1), (0, 1) will be adjacent to (1, 0), and (0, 2) will
        be adjacent to (2, 0). Furthermore, chain the pairs of partition pairs
        together so that, if possible, consecutive pairs [(a, b), (b, a)] of
        partition pairs will share an element in common (i.e. either 'a' or
        'b').
        """

        tuples_of_paired_pairs = create_symmetric_pairs_of_partition_pairs(
            nparts_lhs=nparts_lhs,
            nparts_rhs=nparts_rhs,
        )
        tuples_of_unpaired_pairs = create_unpaired_partition_pairs(
            nparts_lhs=nparts_lhs,
            nparts_rhs=nparts_rhs,
        )

        chained_tuples_of_paired_pairs = \
            chain_tuples_of_paired_pairs(tuples_of_paired_pairs)

        tuples_of_all_pairs = \
            chained_tuples_of_paired_pairs + tuples_of_unpaired_pairs
        flattened_pairs = \
            [pair for _list in tuples_of_all_pairs for pair in _list]

        return torch.FloatTensor(flattened_pairs).int()

    elif bucket_order is BucketOrder.INSIDE_OUT \
            or bucket_order is BucketOrder.OUTSIDE_IN:

        # For bucket_order == 'inside_out', sort the partition pairs so that
        # the first row and first column of tuples are iterated over last,
        # preceded by the second row and second column, etc., such that the very
        # first partition pair is the last row and last column.
        # (Example: if the LHS and RHS each contain 4 partitions, the ordering
        # will be as follows: [
        #   (3, 3),
        #   (2, 2), (2, 3), (3, 2),
        #   (1, 1), (1, 2), (1, 3), (2, 1), (3, 1),
        #   (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0),
        # ]. Note that the partition pairs will be shuffled within each row
        # listed above.)
        #
        # For bucket_order == 'outside_in', sort the partition pairs so that
        # the first row and first column of tuples are iterated over first, then
        # the second row and second column, etc.
        # (Example: if the LHS and RHS each contain 4 partitions, the ordering
        # will be as follows: [
        #   (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0),
        #   (1, 1), (1, 2), (1, 3), (2, 1), (3, 1),
        #   (2, 2), (2, 3), (3, 2),
        #   (3, 3),
        # ]. Note that the partition pairs will be shuffled within each row
        # listed above.)
        #
        # More explicitly, if we imagine the pairs to be arranged in a rectangle
        # with `nparts_lhs` rows and `nparts_rhs` columns, we can sort the pairs
        # in this fashion by dividing up our rectangle into a certain number of
        # L-shaped "layers", the first one of which contains all pairs with
        # either an LHS or RHS indexed by 0, the second one of which contains
        # all remaining pairs with either an LHS or RHS indexed by 1, etc.

        num_layers = min([nparts_lhs, nparts_rhs])
        all_pairs: List[List[int]] = []
        if bucket_order is BucketOrder.INSIDE_OUT:
            layers = range(num_layers - 1, -1, -1)
        elif bucket_order is BucketOrder.OUTSIDE_IN:
            layers = range(num_layers)
        else:
            raise ValueError('Unrecognized bucket_order value!')
        for layer_idx in layers:
            pairs_for_this_layer = create_partition_pairs_for_one_layer(
                nparts_lhs=nparts_lhs,
                nparts_rhs=nparts_rhs,
                layer_idx=layer_idx,
            )
            all_pairs += np.random.permutation(pairs_for_this_layer).tolist()
        return torch.FloatTensor(all_pairs).int()

    elif bucket_order is BucketOrder.RANDOM:

        partition_pairs = torch.stack([
            torch.arange(0, nparts_lhs).expand(nparts_rhs, nparts_lhs).t(),
            torch.arange(0, nparts_rhs).expand(nparts_lhs, nparts_rhs)
        ], dim=2).view(-1, 2)

        # randomize pairs of partitions to iterate over
        return partition_pairs[torch.randperm(partition_pairs.size(0))].int()

    else:
        raise NotImplementedError("Unknown bucket order: %s" % bucket_order)


def create_partition_pairs_for_one_layer(nparts_lhs, nparts_rhs, layer_idx):
    """Create one layer of pairs of tuples to be used by the 'inside_out' and
    'outside_in' modes of create_partition_pairs().
    """

    corner_pair = (layer_idx, layer_idx)
    pairs_forming_row = [
        (layer_idx, column_idx) for column_idx
        in range(layer_idx + 1, nparts_rhs)
    ]
    pairs_forming_column = [
        (row_idx, layer_idx) for row_idx
        in range(layer_idx + 1, nparts_lhs)
    ]
    raw_row_and_column_pairs = [
        val
        for pair in zip_longest(pairs_forming_row, pairs_forming_column)
        for val in pair
    ]
    return [corner_pair] + \
        [pair for pair in raw_row_and_column_pairs if pair is not None]


def create_symmetric_pairs_of_partition_pairs(nparts_lhs, nparts_rhs):
    """
    For the input numbers of LHS and RHS partitions, return a list of length-2
    tuples of all symmetric pairs of partition pairs. (For example, if
    nparts_lhs == 3 and nparts_rhs == 4, we return [
        ((0, 1), (1, 0)),
        ((0, 2), (2, 0)),
        ((1, 2), (2, 1)),
    ].)
    """

    min_length = min([nparts_lhs, nparts_rhs])
    all_tuples = []
    for high_idx in range(min_length):
        for low_idx in range(high_idx):
            all_tuples.append(((low_idx, high_idx), (high_idx, low_idx)))
    return all_tuples


def create_unpaired_partition_pairs(nparts_lhs, nparts_rhs):
    """
    For the input numbers of LHS and RHS partitions, return a list of length-1
    tuples of all partition pairs (a, b) that have no inverse (b, a). (For
    example, if nparts_lhs == 3 and nparts_rhs == 4, we return [
        ((0, 0)),
        ((0, 3)),
        ((1, 1)),
        ((1, 3)),
        ((2, 2)),
        ((2, 3)),
    ].)
    """

    all_tuples = []
    for lhs_idx in range(nparts_lhs):
        for rhs_idx in range(nparts_rhs):
            if (
                lhs_idx == rhs_idx or
                lhs_idx >= nparts_rhs or rhs_idx >= nparts_lhs
            ):
                all_tuples.append(((lhs_idx, rhs_idx),))
    return all_tuples


def chain_tuples_of_paired_pairs(tuples_of_paired_pairs):
    """
    Chain the input pairs of partition pairs together so that, if possible,
    consecutive pairs [(a, b), (b, a)] of partition pairs will share an element
    in common (i.e. either 'a' or 'b').
    """

    if len(tuples_of_paired_pairs) == 0:

        return tuples_of_paired_pairs

    else:

        # Step 1: Shuffle the list of 2-ples of partition pairs and pop off the
        # last element to be the first element in the list of chained 2-ples
        random.shuffle(tuples_of_paired_pairs)
        chained_tuples = [tuples_of_paired_pairs.pop()]

        # Step 2: While there are still partition pairs in the original shuffled
        # list of 2-ples, go through the list and see if there exists a
        # remaining 2-ple ((c,d), (d,c)) such that, if the current last 2-ple in
        # the new *chained* list is ((a,b), (b,a)), either c or d is equal to
        # either a or b. (If not, default to popping the last partition pair
        # from the original shuffled list and appending it to the new chained
        # list.)
        # NOTE: this is an O(n^2) operation, where n is the number of elements
        # in the original shuffled list of partition pairs. Does this need to be
        # sped up (for instance, if we think we'll eventually have a very large
        # number of partition pairs)?
        while len(tuples_of_paired_pairs) > 0:

            first_pair_of_last_chained_tuple = chained_tuples[-1][0]
            # We only need to look at the first pair (a,b) of the last chained
            # tuple ((a,b), (b,a)), since the tuple itself consists of two
            # symmetric tuples

            for idx, candidate_tuple in enumerate(tuples_of_paired_pairs):

                first_pair_of_candidate_tuple = candidate_tuple[0]
                # The same reasoning applies here as in the comment above

                if len([
                    True for element in first_pair_of_candidate_tuple
                    if element in first_pair_of_last_chained_tuple
                ]) > 0:
                    chained_tuples.append(tuples_of_paired_pairs.pop(idx))
                    break

            else:

                # There are no candidate tuples in the original shuffled list
                # with an element that matches the end of the current list of
                # chained tuples, so just default to adding the last element of
                # the original shuffled list to the current list of chained
                # tuples
                chained_tuples.append(tuples_of_paired_pairs.pop())

        return chained_tuples


# compute a randomized AUC using a fixed number of sample points
# NOTE: AUC is the probability that a randomly chosen positive example
# has a higher score than a randomly chosen negative example
def compute_randomized_auc(pos_, neg_, num_samples):
    pos_, neg_ = pos_.view(-1), neg_.view(-1)
    diff = pos_[torch.randint(len(pos_), (num_samples,))] \
           > neg_[torch.randint(len(neg_), (num_samples,))]
    return float(diff.float().mean())


# makes it easy to pass around process group args as a dict and pass as kwargs
def init_process_group(backend='gloo',
                       init_method=None,
                       world_size=None,
                       rank=None,
                       groups=None,
                       **kwargs):
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
