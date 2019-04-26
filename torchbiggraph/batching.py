#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch

from .entitylist import EntityList
from .model import MultiRelationEmbedder
from .stats import Stats
from .types import LongTensorType


def group_by_relation_type(
    rel: LongTensorType,
    lhs: EntityList,
    rhs: EntityList,
) -> List[Tuple[EntityList, EntityList, int]]:
    """Split the edge list in groups that have the same relation type."""
    result: List[Tuple[EntityList, EntityList, int]] = []

    if rel.size(0) == 0:
        return result

    # FIXME Is PyTorch's sort stable? Won't this risk messing up the random shuffle?
    sorted_rel, order = rel.sort()
    delta = sorted_rel[1:] - sorted_rel[:-1]
    cutpoints = (delta.nonzero().flatten() + 1).tolist()

    sorted_lhs = lhs[order]
    sorted_rhs = rhs[order]

    for start, end in zip([0] + cutpoints, cutpoints + [rel.size(0)]):
        result.append((sorted_lhs[start:end],
                       sorted_rhs[start:end],
                       sorted_rel[start].item()))

    return result


def batch_edges_mix_relation_types(
    lhs: EntityList,
    rhs: EntityList,
    rel: LongTensorType,
    *,
    batch_size: int,
) -> Iterable[Tuple[EntityList, EntityList, LongTensorType]]:
    """Split the edges in batches that can contain multiple relation types

    The output preserves the input's order. Batches are all of the same size,
    except possibly the last one.
    """
    for offset in range(0, rel.size(0), batch_size):
        batch_lhs = lhs[offset:offset + batch_size]
        batch_rhs = rhs[offset:offset + batch_size]
        batch_rel = rel[offset:offset + batch_size]
        yield batch_lhs, batch_rhs, batch_rel


def batch_edges_group_by_relation_type(
    lhs: EntityList,
    rhs: EntityList,
    rel: LongTensorType,
    *,
    batch_size: int,
) -> Iterable[Tuple[EntityList, EntityList, int]]:
    """Split the edges in batches that each contain a single relation type

    Batches are all of the same size, except possibly the last one for each
    relation type.
    """
    edges_by_rel = group_by_relation_type(rel, lhs, rhs)
    num_edges_left_by_rel = torch.tensor(
        [lhs.size(0) for lhs, _, _ in edges_by_rel], dtype=torch.long)

    while num_edges_left_by_rel.sum() > 0:
        rel_idx = torch.multinomial(num_edges_left_by_rel.float(), 1).item()
        lhs_rel, rhs_rel, rel_type = edges_by_rel[rel_idx]
        offset = lhs_rel.size(0) - num_edges_left_by_rel[rel_idx].item()
        batch_lhs = lhs_rel[offset:offset + batch_size]
        batch_rhs = rhs_rel[offset:offset + batch_size]
        yield batch_lhs, batch_rhs, rel_type
        num_edges_left_by_rel[rel_idx] -= batch_lhs.size(0)


def call(f: Callable[[], Stats]) -> Stats:
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()


def process_in_batches(
    batch_size: int,
    model: MultiRelationEmbedder,
    batch_processor: "AbstractBatchProcessor",
    lhs: EntityList,
    rhs: EntityList,
    rel: LongTensorType,
    indices: Optional[Union[slice, LongTensorType]] = None,
    delay: float = 0.0,
) -> Stats:
    """Split lhs, rhs and rel in batches, process them and sum the stats

    If indices is not None, only operate on x[indices] for x = lhs, rhs and rel.
    If delay is positive, wait for that many seconds before starting.
    """
    if indices is not None:
        lhs = lhs[indices]
        rhs = rhs[indices]
        rel = rel[indices]

    time.sleep(delay)

    # FIXME: it's not really safe to do partial batches if num_batch_negs != 0
    # because partial batches will produce incorrect results, and if the
    # dataset per thread is very small then every batch may be partial. I don't
    # know of a perfect solution for this that doesn't introduce other biases...

    all_stats = []

    if model.num_dynamic_rels > 0:
        for batch_lhs, batch_rhs, batch_rel in batch_edges_mix_relation_types(
            lhs, rhs, rel, batch_size=batch_size
        ):
            all_stats.append(batch_processor.process_one_batch(
                model, batch_lhs, batch_rhs, batch_rel))
    else:
        for batch_lhs, batch_rhs, rel_type in batch_edges_group_by_relation_type(
            lhs, rhs, rel, batch_size=batch_size
        ):
            all_stats.append(batch_processor.process_one_batch(
                model, batch_lhs, batch_rhs, rel_type))

    stats = Stats.sum(all_stats)
    if isinstance(indices, torch.LongTensor):
        assert stats.count == indices.size(0)
    return stats


class AbstractBatchProcessor(ABC):

    @abstractmethod
    def process_one_batch(
        self,
        model: MultiRelationEmbedder,
        batch_lhs: EntityList,
        batch_rhs: EntityList,
        # batch_rel is int in normal mode, LongTensor in dynamic relations mode.
        batch_rel: Union[int, LongTensorType],
    ) -> Stats:
        pass
