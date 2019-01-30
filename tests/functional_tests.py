#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, NamedTuple, Tuple
from unittest import TestCase, main

import attr
import h5py
import torch
from torch.distributions.multinomial import Multinomial

from torchbiggraph.config import EntitySchema, RelationSchema, ConfigSchema
from torchbiggraph.train import train
from torchbiggraph.eval import do_eval


class Dataset(NamedTuple):
    entity_path: TemporaryDirectory
    relation_paths: List[TemporaryDirectory]

    def cleanup(self):
        self.entity_path.cleanup()
        for path in self.relation_paths:
            path.cleanup()


def broadcast_nums(nums: Iterable[int]) -> int:
    different_nums = set(nums) - {1}
    if len(different_nums) > 1:
        raise RuntimeError("%s cannot be broadcast to a single value" % nums)
    return different_nums.pop() if different_nums else 1


def generate_dataset(
    config: ConfigSchema,
    num_entities: int,
    fractions: List[float],
) -> Dataset:
    """Create a randomly-generated dataset compatible with the given config.

    Create embeddings for each entity (generating the same given number of
    entities for each type) and produce an edge between them if their dot
    product is positive. The edges are split up into several sets, each one
    containing a fraction of the total, as given in the argument (the fractions
    can sum to less than 1, in which case the leftover edges are discarded).

    """
    entity_path = TemporaryDirectory()
    relation_paths = [TemporaryDirectory() for _ in fractions]

    embeddings: Dict[str, Tuple[torch.FloatTensor]] = {}
    for entity_name, entity in config.entities.items():
        embeddings[entity_name] = torch.split(
            torch.randn(num_entities, config.dimension),
            Multinomial(
                num_entities, torch.ones(entity.numPartitions)
            ).sample().long().tolist(),
        )
        for partition, embedding in enumerate(embeddings[entity_name]):
            with open(os.path.join(
                entity_path.name, "entity_count_%s_%d.pt" % (entity_name, partition + 1)
            ), "wb") as f:
                torch.save(len(embedding), f)

    num_lhs_partitions = \
        broadcast_nums(len(embeddings[relation.lhs]) for relation in config.relations)
    num_rhs_partitions = \
        broadcast_nums(len(embeddings[relation.rhs]) for relation in config.relations)

    for lhs_partition in range(num_lhs_partitions):
        for rhs_partition in range(num_rhs_partitions):
            edges = torch.empty(0, 3, dtype=torch.long)
            for rel_idx, relation in enumerate(config.relations):
                scores = torch.einsum(
                    'ld,rd->lr',
                    embeddings[relation.lhs][lhs_partition],
                    embeddings[relation.rhs][rhs_partition],
                )
                these_edges = torch.nonzero(scores > 0)
                edges = torch.cat([
                    edges,
                    torch.cat([
                        these_edges,
                        torch.full((len(these_edges), 1), rel_idx, dtype=torch.long),
                    ], dim=1),
                ], dim=0)
            edges = edges[torch.randperm(len(edges))]
            start_idx = 0
            for fraction, path in zip(fractions, relation_paths):
                end_idx = start_idx + int(fraction * len(edges))
                with h5py.File(os.path.join(
                    path.name, "edges_%d_%d.h5" % (lhs_partition + 1, rhs_partition + 1)
                )) as hf:
                    # Adding one because of Lua indexing.
                    hf["lhs"] = edges[start_idx:end_idx, 0] + 1
                    hf["rhs"] = edges[start_idx:end_idx, 1] + 1
                    hf["rel"] = edges[start_idx:end_idx, 2] + 1
                start_idx = end_idx

    return Dataset(entity_path, relation_paths)


class TestFunctional(TestCase):

    def setUp(self):
        self.outdir = TemporaryDirectory()
        self.addCleanup(self.outdir.cleanup)

    def test_default(self):
        entity_name = "e"
        relation_config = RelationSchema(
            name="r", lhs=entity_name, rhs=entity_name)
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(numPartitions=1)},
            entityPath=None,  # filled in later
            edgePaths=[],  # filled in later
            outdir=self.outdir.name,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4, 0.2]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entityPath=dataset.entity_path.name,
            edgePaths=[dataset.relation_paths[0].name],
        )
        eval_config = attr.evolve(
            base_config,
            entityPath=dataset.entity_path.name,
            edgePaths=[dataset.relation_paths[1].name],
            relations=[attr.evolve(relation_config, all_rhs_negs=1)],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        train(train_config, rank=0)
        do_eval(eval_config)


if __name__ == '__main__':
    main()
