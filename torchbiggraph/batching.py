#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.losses import AbstractLossFunction
from torchbiggraph.model import MultiRelationEmbedder, Scores, override_model
from torchbiggraph.stats import Stats
from torchbiggraph.types import LongTensorType


def group_by_relation_type(edges: EdgeList) -> List[EdgeList]:
    """Split the edge list in groups that have the same relation type."""
    if len(edges) == 0:
        return []
    if edges.has_scalar_relation_type():
        return [edges]

    # FIXME Is PyTorch's sort stable? Won't this risk messing up the random shuffle?
    sorted_rel, order = edges.rel.sort()
    delta = sorted_rel[1:] - sorted_rel[:-1]
    cutpoints = (delta.nonzero().flatten() + 1).tolist()

    result: List[EdgeList] = []
    for start, end in zip([0] + cutpoints, cutpoints + [len(edges)]):
        rel_type = sorted_rel[start]
        edges_for_rel_type = edges[order[start:end]]
        result.append(
            EdgeList(edges_for_rel_type.lhs, edges_for_rel_type.rhs, rel_type)
        )
    return result


def batch_edges_mix_relation_types(
    edges: EdgeList, *, batch_size: int
) -> Iterable[EdgeList]:
    """Split the edges in batches that can contain multiple relation types

    The output preserves the input's order. Batches are all of the same size,
    except possibly the last one.
    """
    for offset in range(0, len(edges), batch_size):
        yield edges[offset : offset + batch_size]


def batch_edges_group_by_relation_type(
    edges: EdgeList, *, batch_size: int
) -> Iterable[EdgeList]:
    """Split the edges in batches that each contain a single relation type

    Batches are all of the same size, except possibly the last one for each
    relation type.
    """
    edge_groups = group_by_relation_type(edges)
    num_edges_left_per_group = torch.tensor(
        [len(edges) for edges in edge_groups], dtype=torch.long
    )

    while num_edges_left_per_group.sum() > 0:
        idx = int(torch.multinomial(num_edges_left_per_group.float(), 1))
        edge_group = edge_groups[idx]
        offset = len(edge_group) - int(num_edges_left_per_group[idx])
        sub_edges = edge_group[offset : offset + batch_size]
        yield sub_edges
        num_edges_left_per_group[idx] -= len(sub_edges)


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
    edges: EdgeList,
    indices: Optional[LongTensorType] = None,
    delay: float = 0.0,
) -> Stats:
    """Split lhs, rhs and rel in batches, process them and sum the stats

    If indices is not None, only operate on x[indices] for x = lhs, rhs and rel.
    If delay is positive, wait for that many seconds before starting.
    """
    if indices is not None:
        edges = edges[indices]

    time.sleep(delay)

    # FIXME: it's not really safe to do partial batches if num_batch_negs != 0
    # because partial batches will produce incorrect results, and if the
    # dataset per thread is very small then every batch may be partial. I don't
    # know of a perfect solution for this that doesn't introduce other biases...

    all_stats = []

    if model.num_dynamic_rels > 0:
        batcher = batch_edges_mix_relation_types
    else:
        batcher = batch_edges_group_by_relation_type

    for batch_edges in batcher(edges, batch_size=batch_size):
        all_stats.append(batch_processor.process_one_batch(model, batch_edges))

    stats = Stats.sum(all_stats)
    return stats


class AbstractBatchProcessor(ABC):
    def __init__(
        self,
        loss_fn: AbstractLossFunction,
        relation_weights: List[float],
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.loss_fn = loss_fn
        self.relation_weights = relation_weights
        self.overrides = overrides

    def calc_loss(self, scores: Scores, batch_edges: EdgeList):

        lhs_loss = self.loss_fn(scores.lhs_pos, scores.lhs_neg)
        rhs_loss = self.loss_fn(scores.rhs_pos, scores.rhs_neg)
        relation = (
            batch_edges.get_relation_type_as_scalar()
            if batch_edges.has_scalar_relation_type()
            else 0
        )
        loss = self.relation_weights[relation] * (lhs_loss + rhs_loss)

        return loss

    @abstractmethod
    def _process_one_batch(
        self, model: MultiRelationEmbedder, batch_edges: EdgeList
    ) -> Stats:
        pass

    def process_one_batch(
        self, model: MultiRelationEmbedder, batch_edges: EdgeList
    ) -> Stats:
        if self.overrides is not None:
            with override_model(model, **self.overrides):
                return self._process_one_batch(model, batch_edges)
        else:
            return self._process_one_batch(model, batch_edges)
