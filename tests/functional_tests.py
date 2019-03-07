#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, NamedTuple, Tuple
from unittest import TestCase, main

import attr
import h5py
import numpy as np

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

    embeddings: Dict[str, Tuple[np.ndarray]] = {}
    for entity_name, entity in config.entities.items():
        embeddings[entity_name] = np.split(
            np.random.randn(num_entities, config.dimension),
            np.cumsum(np.random.multinomial(
                num_entities, [1 / entity.num_partitions] * entity.num_partitions,
            )[:-1]),
        )
        for partition, embedding in enumerate(embeddings[entity_name]):
            with open(os.path.join(
                entity_path.name, "entity_count_%s_%d.txt" % (entity_name, partition)
            ), "xt") as tf:
                tf.write("%d" % len(embedding))

    any_lhs_featurized = any(
        config.entities[relation.lhs].featurized for relation in config.relations)
    any_rhs_featurized = any(
        config.entities[relation.rhs].featurized for relation in config.relations)
    num_lhs_partitions = \
        broadcast_nums(len(embeddings[relation.lhs]) for relation in config.relations)
    num_rhs_partitions = \
        broadcast_nums(len(embeddings[relation.rhs]) for relation in config.relations)

    for lhs_partition in range(num_lhs_partitions):
        for rhs_partition in range(num_rhs_partitions):
            dtype = [("lhs", np.int64), ("lhs_feat", np.bool_),
                     ("rhs", np.int64), ("rhs_feat", np.bool_),
                     ("rel", np.int64)]
            edges = np.empty((0,), dtype=dtype)
            for rel_idx, relation in enumerate(config.relations):
                lhs_partitioned = config.entities[relation.lhs].num_partitions > 1
                rhs_partitioned = config.entities[relation.rhs].num_partitions > 1
                lhs_embs = embeddings[relation.lhs][lhs_partition if lhs_partitioned else 0]
                rhs_embs = embeddings[relation.rhs][rhs_partition if rhs_partitioned else 0]
                scores = np.einsum('ld,rd->lr', lhs_embs, rhs_embs)
                num_these_edges = np.count_nonzero(scores > 0)
                these_edges = np.empty(num_these_edges, dtype=dtype)
                these_edges["lhs"], these_edges["rhs"] = np.nonzero(scores > 0)
                these_edges["rel"] = rel_idx
                these_edges["lhs_feat"] = config.entities[relation.lhs].featurized
                these_edges["rhs_feat"] = config.entities[relation.rhs].featurized
                edges = np.append(edges, these_edges)
            edges = edges[np.random.permutation(len(edges))]
            start_idx = 0
            for fraction, path in zip(fractions, relation_paths):
                end_idx = start_idx + int(fraction * len(edges))
                with h5py.File(os.path.join(
                    path.name, "edges_%d_%d.h5" % (lhs_partition, rhs_partition)
                ), "x") as hf:
                    hf.attrs["format_version"] = 1
                    these_edges = edges[start_idx:end_idx]
                    if any_lhs_featurized:
                        hf["lhsd_data"] = these_edges["lhs"][these_edges["lhs_feat"]]
                        hf["lhsd_offsets"] = np.concatenate((
                            np.array([0], dtype=np.int64),
                            np.cumsum(these_edges["lhs_feat"], dtype=np.int64)))
                        # Poison the non-featurized data.
                        these_edges["lhs"][these_edges["lhs_feat"]] = -1
                    if any_rhs_featurized:
                        hf["rhsd_data"] = these_edges["rhs"][these_edges["rhs_feat"]]
                        hf["rhsd_offsets"] = np.concatenate((
                            np.array([0], dtype=np.int64),
                            np.cumsum(these_edges["rhs_feat"], dtype=np.int64)))
                        # Poison the non-featurized data.
                        these_edges["rhs"][these_edges["rhs_feat"]] = -1
                    hf["lhs"] = these_edges["lhs"]
                    hf["rhs"] = these_edges["rhs"]
                    hf["rel"] = these_edges["rel"]
                start_idx = end_idx

    return Dataset(entity_path, relation_paths)


def init_embeddings(
    target: str,
    config: ConfigSchema,
    *,
    version: int = 0,
):
    with open(os.path.join(target, "checkpoint_version.txt"), "xt") as tf:
        tf.write("%d" % version)
    for entity_name, entity in config.entities.items():
        for partition in range(entity.num_partitions):
            with open(os.path.join(
                config.entity_path,
                "entity_count_%s_%d.txt" % (entity_name, partition),
            ), "rt") as tf:
                entity_count = int(tf.read().strip())
            with h5py.File(os.path.join(
                target,
                "embeddings_%s_%d.v%d.h5" % (entity_name, partition, version),
            ), "x") as hf:
                hf.attrs["format_version"] = 1
                hf.create_dataset("embeddings",
                                  data=np.random.randn(entity_count, config.dimension))
    with h5py.File(os.path.join(target, "model.v%d.h5" % version), "x") as hf:
        hf.attrs["format_version"] = 1


class TestFunctional(TestCase):

    def setUp(self):
        self.checkpoint_path = TemporaryDirectory()
        self.addCleanup(self.checkpoint_path.cleanup)

    def assertCheckpointWritten(self, config: ConfigSchema, *, version: int):
        with open(os.path.join(config.checkpoint_path, "checkpoint_version.txt"), "rt") as tf:
            self.assertEqual(version, int(tf.read().strip()))

        with open(os.path.join(config.checkpoint_path, "config.json"), "rt") as tf:
            self.assertEqual(json.load(tf), config.to_dict())

        self.assertTrue(os.path.exists(os.path.join(
            config.checkpoint_path, "model.v%d.h5" % version)))

        for entity_name, entity in config.entities.items():
            for partition in range(entity.num_partitions):
                with open(os.path.join(
                    config.entity_path,
                    "entity_count_%s_%d.txt" % (entity_name, partition),
                ), "rt") as tf:
                    entity_count = int(tf.read().strip())
                with h5py.File(os.path.join(
                    config.checkpoint_path,
                    "embeddings_%s_%d.v%d.h5" % (entity_name, partition, version),
                ), "r") as hf:
                    embeddings_dataset = hf["embeddings"]
                    self.assertEqual(embeddings_dataset.dtype, np.float32)
                    self.assertEqual(embeddings_dataset.shape,
                                     (entity_count, config.dimension))

    def test_default(self):
        entity_name = "e"
        relation_config = RelationSchema(
            name="r", lhs=entity_name, rhs=entity_name)
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(num_partitions=1)},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4, 0.2]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        eval_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[1].name],
            relations=[attr.evolve(relation_config, all_negs=True)],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        train(train_config, rank=0)
        do_eval(eval_config)
        self.assertCheckpointWritten(train_config, version=1)

    def test_resume_from_checkpoint(self):
        entity_name = "e"
        relation_config = RelationSchema(
            name="r", lhs=entity_name, rhs=entity_name)
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(num_partitions=1)},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
            num_epochs=2,
            num_edge_chunks=2,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4, 0.4]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[d.name for d in dataset.relation_paths],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        init_embeddings(train_config.checkpoint_path, train_config, version=7)
        train(train_config, rank=0)
        self.assertCheckpointWritten(train_config, version=8)
        # Check we did resume the run, not start the whole thing anew.
        self.assertFalse(os.path.exists(
            os.path.join(train_config.checkpoint_path, "model.v6.h5")))

    def test_with_initial_value(self):
        entity_name = "e"
        relation_config = RelationSchema(
            name="r", lhs=entity_name, rhs=entity_name)
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(num_partitions=1)},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4]
        )
        self.addCleanup(dataset.cleanup)
        init_dir = TemporaryDirectory()
        self.addCleanup(init_dir.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
            init_path=init_dir.name,
        )
        # Just make sure no exceptions are raised and nothing crashes.
        init_embeddings(train_config.init_path, train_config)
        train(train_config, rank=0)
        self.assertCheckpointWritten(train_config, version=1)

    def test_featurized(self):
        e1 = EntitySchema(num_partitions=1, featurized=True)
        e2 = EntitySchema(num_partitions=1)
        r1 = RelationSchema(name="r1", lhs="e1", rhs="e2")
        r2 = RelationSchema(name="r2", lhs="e2", rhs="e1")
        base_config = ConfigSchema(
            dimension=10,
            relations=[r1, r2],
            entities={"e1": e1, "e2": e2},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4, 0.2]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        eval_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[1].name],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        train(train_config, rank=0)
        do_eval(eval_config)
        self.assertCheckpointWritten(train_config, version=1)

    def test_partitioned(self):
        e1 = EntitySchema(num_partitions=1)
        e2 = EntitySchema(num_partitions=2)
        e3 = EntitySchema(num_partitions=3)
        r1 = RelationSchema(name="r1", lhs="e1", rhs="e3")
        r2 = RelationSchema(name="r2", lhs="e2", rhs="e3")
        r3 = RelationSchema(name="r3", lhs="e2", rhs="e1")
        base_config = ConfigSchema(
            dimension=10,
            relations=[r1, r2, r3],
            entities={"e1": e1, "e2": e2, "e3": e3},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4, 0.2]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        eval_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[1].name],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        train(train_config, rank=0)
        do_eval(eval_config)
        self.assertCheckpointWritten(train_config, version=1)


if __name__ == '__main__':
    main()
