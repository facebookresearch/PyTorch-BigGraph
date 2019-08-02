#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import json
import logging
import multiprocessing as mp
import os.path
import random
import time
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, NamedTuple, Tuple
from unittest import TestCase, main

import attr
import h5py
import numpy as np

from torchbiggraph.config import (
    ConfigSchema,
    EntitySchema,
    RelationSchema,
)
from torchbiggraph.eval import do_eval
from torchbiggraph.partitionserver import run_partition_server
from torchbiggraph.train import train


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

    def setUp(self) -> None:
        self.checkpoint_path = TemporaryDirectory()
        self.addCleanup(self.checkpoint_path.cleanup)

        seed = random.getrandbits(32)
        np.random.seed(seed)
        logging.info("Random seed: %s", seed)

    def assertHasMetadata(self, hf: h5py.File, config: ConfigSchema) -> None:
        self.assertEqual(hf.attrs["format_version"], 1)
        self.assertEqual(json.loads(hf.attrs["config/json"]), config.to_dict())
        self.assertCountEqual(
            [key.partition("/")[-1]
             for key in hf.attrs.keys()
             if key.startswith("iteration/")],
            ["num_epochs", "epoch_idx",
             "num_edge_paths", "edge_path_idx", "edge_path",
             "num_edge_chunks", "edge_chunk_idx"])

    def assertIsModelParameter(self, dataset: h5py.Dataset) -> None:
        # In fact it could also be a group...
        if not isinstance(dataset, h5py.Dataset):
            return
        self.assertIn("state_dict_key", dataset.attrs)
        self.assertTrue(np.isfinite(dataset[...]).all())

    def assertIsModelParameters(self, group: h5py.Group) -> None:
        self.assertIsInstance(group, h5py.Group)
        group.visititems(lambda _, d: self.assertIsModelParameter(d))

    def assertIsOptimStateDict(self, dataset: h5py.Dataset) -> None:
        self.assertIsInstance(dataset, h5py.Dataset)
        self.assertEqual(dataset.dtype, np.dtype("V1"))
        self.assertEqual(len(dataset.shape), 1)

    def assertIsEmbeddings(
        self,
        dataset: h5py.Dataset,
        entity_count: int,
        dimension: int,
    ) -> None:
        self.assertIsInstance(dataset, h5py.Dataset)
        self.assertEqual(dataset.dtype, np.float32)
        self.assertEqual(dataset.shape, (entity_count, dimension))
        self.assertTrue(np.all(np.isfinite(dataset[...])))
        self.assertTrue(np.all(np.linalg.norm(dataset[...], axis=-1) != 0))

    def assertCheckpointWritten(self, config: ConfigSchema, *, version: int) -> None:
        with open(os.path.join(config.checkpoint_path, "checkpoint_version.txt"), "rt") as tf:
            self.assertEqual(version, int(tf.read().strip()))

        with open(os.path.join(config.checkpoint_path, "config.json"), "rt") as tf:
            self.assertEqual(json.load(tf), config.to_dict())

        with h5py.File(os.path.join(
            config.checkpoint_path, "model.v%d.h5" % version
        ), "r") as hf:
            self.assertHasMetadata(hf, config)
            self.assertIsModelParameters(hf["model"])
            self.assertIsOptimStateDict(hf["optimizer/state_dict"])

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
                    self.assertHasMetadata(hf, config)
                    self.assertIsEmbeddings(
                        hf["embeddings"], entity_count, config.dimension)
                    self.assertIsOptimStateDict(hf["optimizer/state_dict"])

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
            workers=2,
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
        self.assertCheckpointWritten(train_config, version=1)
        do_eval(eval_config)

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
            workers=2,
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
            workers=2,
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
            workers=2,
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
        self.assertCheckpointWritten(train_config, version=1)
        do_eval(eval_config)

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
            workers=2,
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
        self.assertCheckpointWritten(train_config, version=1)
        do_eval(eval_config)

    def test_distributed(self):
        sync_path = TemporaryDirectory()
        self.addCleanup(sync_path.cleanup)
        entity_name = "e"
        relation_config = RelationSchema(
            name="r",
            lhs=entity_name,
            rhs=entity_name,
            operator="linear",  # To exercise the parameter server.
        )
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(num_partitions=4)},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
            num_machines=2,
            distributed_init_method="file://%s" % os.path.join(sync_path.name, "sync"),
            workers=2,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        trainer0 = mp.get_context("spawn").Process(
            name="trainer#0",
            target=train, args=(train_config,), kwargs={"rank": 0})
        trainer1 = mp.get_context("spawn").Process(
            name="trainer#1",
            target=train, args=(train_config,), kwargs={"rank": 1})
        # FIXME In Python 3.7 use kill here.
        self.addCleanup(trainer0.terminate)
        self.addCleanup(trainer1.terminate)
        trainer0.start()
        trainer1.start()
        done = [False, False]
        while not all(done):
            time.sleep(1)
            if not trainer0.is_alive() and not done[0]:
                self.assertEqual(trainer0.exitcode, 0)
                done[0] = True
            if not trainer1.is_alive() and not done[1]:
                self.assertEqual(trainer1.exitcode, 0)
                done[1] = True
        self.assertCheckpointWritten(train_config, version=1)

    def test_distributed_with_partition_servers(self):
        sync_path = TemporaryDirectory()
        self.addCleanup(sync_path.cleanup)
        entity_name = "e"
        relation_config = RelationSchema(
            name="r", lhs=entity_name, rhs=entity_name)
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={entity_name: EntitySchema(num_partitions=4)},
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
            num_machines=2,
            num_partition_servers=1,
            distributed_init_method="file://%s" % os.path.join(sync_path.name, "sync"),
            workers=2,
        )
        dataset = generate_dataset(
            base_config, num_entities=100, fractions=[0.4]
        )
        self.addCleanup(dataset.cleanup)
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        trainer0 = mp.get_context("spawn").Process(
            name="trainer#0",
            target=train, args=(train_config,), kwargs={"rank": 0})
        trainer1 = mp.get_context("spawn").Process(
            name="trainer#1",
            target=train, args=(train_config,), kwargs={"rank": 1})
        partition_server = mp.get_context("spawn").Process(
            name="partition server#0",
            target=run_partition_server, args=(train_config,), kwargs={"rank": 0})
        # FIXME In Python 3.7 use kill here.
        self.addCleanup(trainer0.terminate)
        self.addCleanup(trainer1.terminate)
        self.addCleanup(partition_server.terminate)
        trainer0.start()
        trainer1.start()
        partition_server.start()
        done = [False, False]
        while not all(done):
            time.sleep(1)
            if not trainer0.is_alive() and not done[0]:
                self.assertEqual(trainer0.exitcode, 0)
                done[0] = True
            if not trainer1.is_alive() and not done[1]:
                self.assertEqual(trainer1.exitcode, 0)
                done[1] = True
            if not partition_server.is_alive():
                self.fail("Partition server died with exit code %d"
                          % partition_server.exitcode)
        partition_server.terminate()  # Cannot be shut down gracefully.
        partition_server.join()
        logging.info("Partition server died with exit code %d",
                     partition_server.exitcode)
        self.assertCheckpointWritten(train_config, version=1)

    def test_dynamic_relations(self):
        relation_config = RelationSchema(name="r", lhs="el", rhs="er")
        base_config = ConfigSchema(
            dimension=10,
            relations=[relation_config],
            entities={
                "el": EntitySchema(num_partitions=1),
                "er": EntitySchema(num_partitions=1),
            },
            entity_path=None,  # filled in later
            edge_paths=[],  # filled in later
            checkpoint_path=self.checkpoint_path.name,
            dynamic_relations=True,
            global_emb=False,  # Must be off for dynamic relations.
            workers=2,
        )
        gen_config = attr.evolve(
            base_config,
            relations=[relation_config] * 10,
            dynamic_relations=False,  # Must be off if more than 1 relation.
        )
        dataset = generate_dataset(
            gen_config, num_entities=100, fractions=[0.04, 0.02]
        )
        self.addCleanup(dataset.cleanup)
        with open(os.path.join(
            dataset.entity_path.name, "dynamic_rel_count.txt"
        ), "xt") as f:
            f.write("%d" % len(gen_config.relations))
        train_config = attr.evolve(
            base_config,
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[0].name],
        )
        eval_config = attr.evolve(
            base_config,
            relations=[attr.evolve(relation_config, all_negs=True)],
            entity_path=dataset.entity_path.name,
            edge_paths=[dataset.relation_paths[1].name],
        )
        # Just make sure no exceptions are raised and nothing crashes.
        train(train_config, rank=0)
        self.assertCheckpointWritten(train_config, version=1)
        do_eval(eval_config)


if __name__ == '__main__':
    main()
