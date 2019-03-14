#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
import random
import sys
import time
from enum import Enum
from itertools import chain
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import attr
import torch
import torch.distributed as td
import torch.multiprocessing as mp
from torch.optim import Optimizer, Adagrad

from .config import parse_config, ConfigSchema
from .entitylist import EntityList
from .eval import eval_many_batches, EvalStats
from .fileio import CheckpointManager, EdgeReader, MetadataProvider, \
    ConfigMetadataProvider, maybe_old_entity_path
from .lockserver import setup_lock_server
from .model import make_model, override_model, MultiRelationEmbedder
from .parameterserver import setup_parameter_server_thread, \
    setup_parameter_server
from .row_adagrad import RowAdagrad
from .util import log, vlog, chunk_by_index, get_partitioned_types, \
    create_pool, fast_approx_rand, DummyOptimizer, create_ordered_buckets, Side, \
    init_process_group, Bucket, Partition, EntityName, Rank, ModuleStateDict, \
    OptimizerStateDict, split_almost_equally, round_up_to_nearest_multiple
from .stats import Stats, stats


class Action(Enum):
    TRAIN = "train"
    EVAL = "eval"


@stats
class TrainStats(Stats):
    loss: float = attr.ib()
    violators_lhs: int = attr.ib()
    violators_rhs: int = attr.ib()


def train_one_batch(
    model: MultiRelationEmbedder,
    optimizers: List[Optimizer],
    batch_lhs: EntityList,
    batch_rhs: EntityList,
    # batch_rel is int in normal mode, LongTensor in dynamic relations mode.
    batch_rel: Union[int, torch.LongTensor],
) -> TrainStats:
    batch_size = batch_lhs.size(0)
    batch_lhs = batch_lhs.collapse(model.is_featurized(batch_rel, Side.LHS))
    batch_rhs = batch_rhs.collapse(model.is_featurized(batch_rel, Side.RHS))
    model.zero_grad()

    loss, (lhs_margin, rhs_margin), _scores = model(batch_lhs,
                                                    batch_rhs,
                                                    batch_rel)

    stats = TrainStats(
        loss=loss.item(),
        # each row has one column which is the self-interaction, so subtract
        # this from the number of violators
        violators_lhs=lhs_margin.gt(0).long().sum().item(),
        violators_rhs=rhs_margin.gt(0).long().sum().item(),
        count=batch_size)

    loss.backward()
    for optimizer in optimizers:
        optimizer.step()

    return stats


def train_many_batches(
    config: ConfigSchema,
    model: MultiRelationEmbedder,
    optimizers: List[Optimizer],
    lhs: EntityList,
    rhs: EntityList,
    rel: torch.LongTensor,
) -> TrainStats:
    all_stats = []
    if model.num_dynamic_rels > 0:
        # do standard batching
        offset, num_edges = 0, rel.size(0)
        while offset < num_edges:
            batch_size = min(num_edges - offset, config.batch_size)
            all_stats.append(train_one_batch(
                model, optimizers,
                lhs[offset:offset + batch_size],
                rhs[offset:offset + batch_size],
                rel[offset:offset + batch_size]))
            offset += batch_size
    else:
        # group the edges by relation, and only do batches of a single
        # relation type
        _, lhs_chunked, rhs_chunked = chunk_by_index(rel, lhs, rhs)
        edge_count_by_relation = torch.LongTensor([e.nelement() for e in lhs_chunked])
        offset_by_relation = torch.LongTensor([0 for e in lhs_chunked])

        while edge_count_by_relation.sum() > 0:
            # pick which relation to do proportional to number of edges of that type
            batch_rel = torch.multinomial(edge_count_by_relation.float(), 1).item()
            batch_size = min(edge_count_by_relation[batch_rel].item(), config.batch_size)
            offset = offset_by_relation[batch_rel]
            lhs_rel = lhs_chunked[batch_rel][offset:offset + batch_size]
            rhs_rel = rhs_chunked[batch_rel][offset:offset + batch_size]

            all_stats.append(train_one_batch(
                model, optimizers, lhs_rel, rhs_rel, batch_rel))

            edge_count_by_relation[batch_rel] -= batch_size
            offset_by_relation[batch_rel] += batch_size

    return TrainStats.sum(all_stats)


def perform_action_one_thread(
    rank: Rank,
    config: ConfigSchema,
    action: Action,
    model: MultiRelationEmbedder,
    epoch_idx: int,
    lhs: EntityList,
    rhs: EntityList,
    rel: torch.LongTensor,
    my_edges: torch.LongTensor,
    optimizers: Optional[List[Optimizer]] = None,
) -> Union[TrainStats, EvalStats]:
    """ This is the main loop executed by each HOGWILD worker thread.
    """
    lhs = lhs[my_edges]
    rhs = rhs[my_edges]
    rel = rel[my_edges]

    # if rank == 0 or rank == config.workers - 1:
    #     log("Rank %d : chunking edges..." % rank)

    if action is Action.TRAIN:
        if optimizers is None:
            raise ValueError("Need optimizers for training")
        if rank > 0 and epoch_idx == 0:
            time.sleep(config.hogwild_delay)
        stats = train_many_batches(config, model, optimizers, lhs, rhs, rel)
    elif action is Action.EVAL:
        eval_batch_size = round_up_to_nearest_multiple(
            config.batch_size, config.eval_num_batch_negs
        )
        eval_config = attr.evolve(config, batch_size=eval_batch_size)

        with override_model(model,
                            num_uniform_negs=config.eval_num_uniform_negs,
                            num_batch_negs=config.eval_num_batch_negs):
            stats = eval_many_batches(eval_config, model, lhs, rhs, rel)
    else:
        raise NotImplementedError("Unknown action: %s" % action)
    assert stats.count == my_edges.size(0)
    return stats


def distribute_action_among_workers(
    pool: mp.Pool,
    config: ConfigSchema,
    action: Action,
    model: MultiRelationEmbedder,
    epoch_idx: int,
    lhs: EntityList,
    rhs: EntityList,
    rel: torch.LongTensor,
    edge_perm: torch.LongTensor,
    optimizers: Optional[List[Optimizer]] = None
) -> Union[TrainStats, EvalStats]:
    all_stats = pool.starmap(perform_action_one_thread, [
        (Rank(i), config, action, model, epoch_idx, lhs, rhs, rel, edge_perm[s], optimizers)
        for i, s in enumerate(split_almost_equally(len(edge_perm), num_parts=config.workers))
    ])

    if action is action.TRAIN:
        return TrainStats.sum(all_stats).average()
    elif action is action.EVAL:
        return EvalStats.sum(all_stats).average()
    else:
        raise NotImplementedError("Unknown action: %s" % action)


def init_embs(
    entity: EntityName,
    entity_count: int,
    dim: int,
    scale: float,
) -> Tuple[torch.FloatTensor, None]:
    """Initialize embeddings of size entity_count x dim.
    """
    # FIXME: Use multi-threaded instead of fast_approx_rand
    vlog("Initializing %s" % entity)
    return fast_approx_rand(entity_count * dim).view(entity_count, dim).mul_(scale), None


RANK_ZERO = Rank(0)


class IterationManager(MetadataProvider):

    def __init__(
        self,
        num_epochs: int,
        edge_paths: List[str],
        num_edge_chunks: int,
        *,
        iteration_idx: int = 0,
    ) -> None:
        self.num_epochs = num_epochs
        self.edge_paths = edge_paths
        self.num_edge_chunks = num_edge_chunks
        self.iteration_idx = iteration_idx

    @property
    def epoch_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks // self.num_edge_paths

    @property
    def num_edge_paths(self) -> int:
        return len(self.edge_paths)

    @property
    def edge_path_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks % self.num_edge_paths

    @property
    def edge_path(self) -> str:
        return self.edge_paths[self.edge_path_idx]

    @property
    def edge_chunk_idx(self) -> int:
        return self.iteration_idx % self.num_edge_chunks

    def remaining_iterations(self) -> Iterable[Tuple[int, int, int]]:
        while self.epoch_idx < self.num_epochs:
            yield self.epoch_idx, self.edge_path_idx, self.edge_chunk_idx
            self.iteration_idx += 1

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "iteration/num_epochs": self.num_epochs,
            "iteration/epoch_idx": self.epoch_idx,
            "iteration/num_edge_paths": self.num_edge_paths,
            "iteration/edge_path_idx": self.edge_path_idx,
            "iteration/edge_path": self.edge_path,
            "iteration/num_edge_chunks": self.num_edge_chunks,
            "iteration/edge_chunk_idx": self.edge_chunk_idx,
        }


def train_and_report_stats(
    config: ConfigSchema,
    rank: Rank = RANK_ZERO,
) -> Generator[Tuple[int, Optional[EvalStats], TrainStats, Optional[EvalStats]], None, None]:
    """Each epoch/pass, for each partition pair, loads in embeddings and edgelist
    from disk, runs HOGWILD training on them, and writes partitions back to disk.
    """

    if config.verbose > 0:
        import pprint
        pprint.PrettyPrinter().pprint(config.to_dict())

    log("Loading entity counts...")
    if maybe_old_entity_path(config.entity_path):
        log("WARNING: It may be that your entity path contains files using the "
            "old format. See D14241362 for how to update them.")
    entity_counts: Dict[str, List[int]] = {}
    for entity, econf in config.entities.items():
        entity_counts[entity] = []
        for part in range(econf.num_partitions):
            with open(os.path.join(
                config.entity_path, "entity_count_%s_%d.txt" % (entity, part)
            ), "rt") as tf:
                entity_counts[entity].append(int(tf.read().strip()))

    partition_server_ranks = None
    if config.num_machines > 1:
        log("Setup lock server...")
        init_method = config.distributed_init_method
        # N param client threads, N param servers, 1 lock server
        world_size = config.num_machines * 3 + 1  # + config.num_partition_servers

        if config.num_partition_servers > 0:
            world_size += config.num_partition_servers
        elif config.num_partition_servers == -1:  # use machines as partition servers
            world_size += config.num_machines

        barrier_group_ranks = list(range(config.num_machines))
        lock_client = setup_lock_server(
            is_server_node=(rank == 0),
            server_rank=3 * config.num_machines,
            num_clients=config.num_machines,
            init_method=init_method,
            world_size=world_size,
            groups=[barrier_group_ranks])

        log("Setup param server...")

        parameter_client = setup_parameter_server_thread(
            client_rank=config.num_machines * 2 + rank,
            server_rank=config.num_machines + rank,
            all_server_ranks=[config.num_machines + x
                              for x in range(config.num_machines)],
            num_clients=config.num_machines,
            init_method=init_method,
            world_size=world_size,
            groups=[barrier_group_ranks])

        num_partition_servers = config.num_partition_servers
        if config.num_partition_servers == -1:
            setup_parameter_server(server_rank=config.num_machines * 3 + 1 + rank,
                                   num_clients=config.num_machines,
                                   init_method=init_method,
                                   world_size=world_size,
                                   groups=[barrier_group_ranks])
            num_partition_servers = config.num_machines

        partition_server_ranks = range(config.num_machines * 3 + 1,
                                       config.num_machines * 3 + 1 + num_partition_servers)

        groups = init_process_group(init_method=init_method,
                                    world_size=world_size,
                                    rank=rank,
                                    groups=[barrier_group_ranks])
        barrier_group = groups[0]

    # fork early for HOGWILD threads
    log("Creating workers...")
    pool = create_pool(config.workers)

    def make_optimizer(params: Iterable[torch.nn.Parameter], is_emb: bool) -> Optimizer:
        params = list(params)
        if len(params) == 0:
            optimizer = DummyOptimizer()
        elif is_emb:
            optimizer = RowAdagrad(params, lr=config.lr)
        else:
            if config.relation_lr is not None:
                lr = config.relation_lr
            else:
                lr = config.lr
            optimizer = Adagrad(params, lr=lr)
        optimizer.share_memory()
        return optimizer

    # background_io is only supported in single-machine mode
    background_io = config.background_io and config.num_machines == 1

    checkpoint_manager = CheckpointManager(
        config.checkpoint_path,
        background=background_io,
        rank=rank,
        num_machines=config.num_machines,
        partition_server_ranks=partition_server_ranks,
    )
    checkpoint_manager.register_metadata_provider(ConfigMetadataProvider(config))
    checkpoint_manager.write_config(config)

    iteration_manager = IterationManager(
        config.num_epochs, config.edge_paths, config.num_edge_chunks,
        iteration_idx=checkpoint_manager.checkpoint_version)
    checkpoint_manager.register_metadata_provider(iteration_manager)

    if config.init_path is not None:
        loadpath_manager = CheckpointManager(config.init_path)
    else:
        loadpath_manager = None

    def load_embeddings(
        entity: EntityName,
        part: Partition,
        strict: bool = False,
        force_dirty: bool = False,
    ) -> Tuple[torch.nn.Parameter, Optional[OptimizerStateDict]]:
        if strict:
            embs, optim_state = checkpoint_manager.read(entity, part,
                                                        force_dirty=force_dirty)
        else:
            # Strict is only false during the first iteration, because in that
            # case the checkpoint may not contain any data (unless a previous
            # run was resumed) so we fall back on initial values.
            embs, optim_state = checkpoint_manager.maybe_read(entity, part,
                                                              force_dirty=force_dirty)
            if embs is None and loadpath_manager is not None:
                embs, optim_state = loadpath_manager.maybe_read(entity, part)
            if embs is None:
                embs, optim_state = init_embs(entity, entity_counts[entity][part],
                                              config.dimension, config.init_scale)
        assert embs.is_shared()
        return torch.nn.Parameter(embs), optim_state

    # Figure out how many lhs and rhs partitions we need
    nparts_lhs, lhs_partitioned_types = get_partitioned_types(config, Side.LHS)
    nparts_rhs, rhs_partitioned_types = get_partitioned_types(config, Side.RHS)
    vlog("nparts %d %d types %s %s" %
         (nparts_lhs, nparts_rhs, lhs_partitioned_types, rhs_partitioned_types))

    log("Initializing global model...")

    model = make_model(config)

    vlog("Initializing optimizers")
    # FIXME: use SGD with different learning rate?
    optimizers: Dict[Optional[Tuple[EntityName, Partition]], Optimizer] = {
        # The "None" optimizer is for the non-entity-specific parameters.
        None: make_optimizer(model.parameters(), False)
    }

    state_dict, optim_state = checkpoint_manager.maybe_read_model()

    if state_dict is None and loadpath_manager is not None:
        state_dict, optim_state = loadpath_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    if optim_state is not None:
        optimizers[None].load_state_dict(optim_state)

    vlog("Loading unpartitioned entities...")
    for entity, econfig in config.entities.items():
        if econfig.num_partitions == 1:
            embs, optim_state = load_embeddings(entity, Partition(0))
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)
            optimizer = make_optimizer([embs], True)
            if optim_state is not None:
                optimizer.load_state_dict(optim_state)
            optimizers[(entity, Partition(0))] = optimizer

    if config.num_machines > 1:
        # start communicating shared parameters with the parameter server
        added_to_parameter_client: Set[int] = set()
        for k, v in ModuleStateDict(model.state_dict()).items():
            if v._cdata not in added_to_parameter_client:
                added_to_parameter_client.add(v._cdata)
                log("Adding %s (%d params) to parameter server" % (k, v.nelement()))
                parameter_client.set_param(k, v.data)

    strict = False

    def swap_partitioned_embeddings(
        old_b: Optional[Bucket],
        new_b: Optional[Bucket],
    ):
        # 0. given the old and new buckets, construct data structures to keep
        #    track of old and new embedding (entity, part) tuples

        io_bytes = 0
        log("Swapping partitioned embeddings %s %s" % (old_b, new_b))

        types = ([(e, Side.LHS) for e in lhs_partitioned_types]
                 + [(e, Side.RHS) for e in rhs_partitioned_types])
        old_parts = {(e, old_b.get_partition(side)): side
                     for e, side in types if old_b is not None}
        new_parts = {(e, new_b.get_partition(side)): side
                     for e, side in types if new_b is not None}

        to_checkpoint = set(old_parts) - set(new_parts)
        preserved = set(old_parts) & set(new_parts)

        # 1. checkpoint embeddings that will not be used in the next pair
        #
        if old_b is not None:  # there are previous embeddings to checkpoint
            log("Writing partitioned embeddings")
            for entity, part in to_checkpoint:
                side = old_parts[(entity, part)]
                vlog("Checkpointing (%s %d %s)" %
                     (entity, part, side.pick("lhs", "rhs")))
                embs = model.get_embeddings(entity, side)
                optim_key = (entity, part)
                optim_state = OptimizerStateDict(optimizers[optim_key].state_dict())
                io_bytes += embs.nelement() * 4  # ignore optim state
                checkpoint_manager.write(entity, part, embs.detach(), optim_state)
                if optim_key in optimizers:
                    del optimizers[optim_key]
                # these variables are holding large objects; let them be freed
                del embs
                del optim_state

            if config.num_machines > 1:
                lock_client.release_bucket(old_b)

        # 2. copy old embeddings that will be used in the next pair
        #    into a temporary dictionary
        #
        tmp_emb = {x: model.get_embeddings(x[0], old_parts[x]) for x in preserved}

        for entity, _ in types:
            model.clear_embeddings(entity, Side.LHS)
            model.clear_embeddings(entity, Side.RHS)

        if new_b is None:  # there are no new embeddings to load
            return io_bytes

        # 3. load new embeddings into the model/optimizer, either from disk
        #    or the temporary dictionary
        #
        log("Loading entities")
        for entity, side in types:
            part = new_b.get_partition(side)
            part_key = (entity, part)
            if part_key in tmp_emb:
                vlog("Loading (%s, %d) from preserved" % (entity, part))
                embs, optim_state = tmp_emb[part_key], None
            else:
                vlog("Loading (%s, %d)" % (entity, part))

                force_dirty = (config.num_machines > 1
                               and lock_client.check_and_set_dirty(entity, part))
                embs, optim_state = load_embeddings(
                    entity, part, strict=strict, force_dirty=force_dirty)
                io_bytes += embs.nelement() * 4  # ignore optim state

            model.set_embeddings(entity, embs, side)
            tmp_emb[part_key] = embs

            optim_key = (entity, part)
            if optim_key not in optimizers:
                vlog("Resetting optimizer %s" % (optim_key,))
                optimizer = make_optimizer([embs], True)
                if optim_state is not None:
                    vlog("Setting optim state")
                    optimizer.load_state_dict(optim_state)

                optimizers[optim_key] = optimizer

        return io_bytes

    # Start of the main training loop.
    for epoch_idx, edge_path_idx, edge_chunk_idx \
            in iteration_manager.remaining_iterations():
        log("Starting epoch %d / %d edge path %d / %d edge chunk %d / %d" %
            (epoch_idx + 1, iteration_manager.num_epochs,
             edge_path_idx + 1, iteration_manager.num_edge_paths,
             edge_chunk_idx + 1, iteration_manager.num_edge_chunks))
        edge_reader = EdgeReader(iteration_manager.edge_path)
        log("edge_path= %s" % iteration_manager.edge_path)

        buckets = create_ordered_buckets(
            nparts_lhs=nparts_lhs,
            nparts_rhs=nparts_rhs,
            order=config.bucket_order,
            generator=random.Random(),
        )
        total_buckets = len(buckets)

        # Print buckets
        vlog('\nPartition pairs:')
        for bucket in buckets:
            vlog("%s" % (bucket,))
        vlog('')

        if config.num_machines > 1:
            td.barrier(group=barrier_group)
            log("Lock client new epoch...")
            if rank == 0:
                lock_client.new_epoch(buckets,
                                      lock_lhs=len(lhs_partitioned_types) > 0,
                                      lock_rhs=len(rhs_partitioned_types) > 0,
                                      init_tree=(config.distributed_tree_init_order
                                                 and edge_path_idx == 0
                                                 and epoch_idx == 0))
            td.barrier(group=barrier_group)

        remaining = len(buckets)
        cur_b = None
        while remaining > 0:
            old_b = cur_b
            io_time = 0.
            io_bytes = 0
            if config.num_machines > 1:
                cur_b, remaining = lock_client.acquire_pair(rank, maybe_old_bucket=old_b)
                cur_b = Bucket(*cur_b) if cur_b is not None else None
                print('still in queue: %d' % remaining, file=sys.stderr)
                if cur_b is None:
                    if old_b is not None:
                        # if you couldn't get a new pair, release the lock
                        # to prevent a deadlock!
                        tic = time.time()
                        io_bytes += swap_partitioned_embeddings(old_b, None)
                        io_time += time.time() - tic
                    time.sleep(1)  # don't hammer td
                    continue
            else:
                cur_b = buckets.pop(0)
                remaining -= 1

            def log_status(msg, always=False):
                f = log if always else vlog
                f("%s: %s" % (cur_b, msg))

            tic = time.time()

            io_bytes += swap_partitioned_embeddings(old_b, cur_b)

            current_index = \
                (iteration_manager.iteration_idx + 1) * total_buckets - remaining

            if remaining > 0 and background_io:
                assert config.num_machines == 1
                # Ensure the previous bucket finished writing to disk.
                checkpoint_manager.wait_for_marker(current_index - 1)

                log_status("Prefetching")
                next_partition = buckets[0]
                for entity in lhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_partition.lhs)
                for entity in rhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_partition.rhs)

                checkpoint_manager.record_marker(current_index)

            log_status("Loading edges")
            lhs, rhs, rel = edge_reader.read(
                cur_b.lhs, cur_b.rhs, edge_chunk_idx, config.num_edge_chunks)
            num_edges = rel.size(0)
            # this might be off in the case of tensorlist
            io_bytes += (lhs.nelement() + rhs.nelement() + rel.nelement()) * 4

            log_status("Shuffling edges")
            # Fix a seed to get the same permutation every time; have it
            # depend on all and only what affects the set of edges.
            g = torch.Generator()
            g.manual_seed(hash((edge_path_idx, edge_chunk_idx, cur_b.lhs, cur_b.rhs)))

            num_eval_edges = int(num_edges * config.eval_fraction)
            if num_eval_edges > 0:
                edge_perm = torch.randperm(num_edges, generator=g)
                eval_edge_perm = edge_perm[-num_eval_edges:]
                num_edges -= num_eval_edges
                edge_perm = edge_perm[torch.randperm(num_edges)]
            else:
                edge_perm = torch.randperm(num_edges)

            # HOGWILD evaluation before training
            eval_stats_before: Optional[EvalStats] = None
            if num_eval_edges > 0:
                log_status("Waiting for workers to perform evaluation")
                eval_stats_before = distribute_action_among_workers(
                    pool, config,
                    Action.EVAL, model, epoch_idx, lhs, rhs, rel, eval_edge_perm)
                log("stats before %s: %s" % (cur_b, eval_stats_before))

            io_time += time.time() - tic
            tic = time.time()
            # HOGWILD training
            log_status("Waiting for workers to perform training")
            stats = distribute_action_among_workers(
                pool, config,
                Action.TRAIN, model, epoch_idx, lhs, rhs, rel, edge_perm,
                list(optimizers.values()))
            compute_time = time.time() - tic

            log_status(
                "bucket %d / %d : Processed %d edges in %.2f s "
                "( %.2g M/sec ); io: %.2f s ( %.2f MB/sec )" %
                (total_buckets - remaining, total_buckets,
                 lhs.size(0), compute_time, lhs.size(0) / compute_time / 1e6,
                 io_time, io_bytes / io_time / 1e6),
                always=True)
            log_status("%s" % stats, always=True)

            # HOGWILD eval after training
            eval_stats_after: Optional[EvalStats] = None
            if num_eval_edges > 0:
                log_status("Waiting for workers to perform evaluation")
                eval_stats_after = distribute_action_among_workers(
                    pool, config,
                    Action.EVAL, model, epoch_idx, lhs, rhs, rel, eval_edge_perm)
                log("stats after %s: %s" % (cur_b, eval_stats_after))

            # Add train/eval metrics to queue
            yield current_index, eval_stats_before, stats, eval_stats_after

        swap_partitioned_embeddings(cur_b, None)

        # Distributed Processing: all machines can leave the barrier now.
        if config.num_machines > 1:
            td.barrier(barrier_group)

        # Write metadata: for multiple machines, write from rank-0
        log("Finished epoch %d path %d pass %d; checkpointing global state."
            % (epoch_idx + 1, edge_path_idx + 1, edge_chunk_idx + 1))
        log("My rank: %d" % rank)
        if rank == 0:
            for entity, econfig in config.entities.items():
                if econfig.num_partitions == 1:
                    embs = model.get_embeddings(entity, Side.LHS)
                    optimizer = optimizers[(entity, Partition(0))]

                    checkpoint_manager.write(
                        entity, Partition(0),
                        embs.detach(), OptimizerStateDict(optimizer.state_dict()))

            sanitized_state_dict: ModuleStateDict = {}
            for k, v in ModuleStateDict(model.state_dict()).items():
                if k.startswith('lhs_embs') or k.startswith('rhs_embs'):
                    # skipping state that's an entity embedding
                    continue
                sanitized_state_dict[k] = v

            log("Writing metadata...")
            checkpoint_manager.write_model(
                sanitized_state_dict,
                OptimizerStateDict(optimizers[None].state_dict()),
            )

        log("Writing the checkpoint...")
        checkpoint_manager.write_new_version(config)

        if config.num_machines > 1:
            log("Waiting for other workers to write their parts of the checkpoint: rank %d" % rank)
            td.barrier(barrier_group)
            log("All parts of the checkpoint have been written")

        log("Switching to new checkpoint version...")
        checkpoint_manager.switch_to_new_version()

        if config.num_machines > 1:
            log("Waiting for other workers to switch to the new checkpoint version: rank %d" % rank)
            td.barrier(barrier_group)
            log("All workers have switched to the new checkpoint version")

        # After all the machines have finished committing
        # checkpoints, we remove the old checkpoints.
        checkpoint_manager.remove_old_version(config)

        # now we're sure that all partition files exist,
        # so be strict about loading them
        strict = True

    # quiescence
    pool.close()
    pool.join()

    if config.num_machines > 1:
        td.barrier(barrier_group)

    checkpoint_manager.close()
    if loadpath_manager is not None:
        loadpath_manager.close()

    # FIXME join distributed workers (not really necessary)

    log("Exiting")


def train(
    config: ConfigSchema,
    rank: Rank = RANK_ZERO,
) -> None:
    # Create and run the generator until exhaustion.
    for _ in train_and_report_stats(config, rank):
        pass


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--rank', type=int, default=0,
                        help="For multi-machine, this machine's rank")
    opt = parser.parse_args()

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
    else:
        overrides = None
    config = parse_config(opt.config, overrides)

    train(config, rank=Rank(opt.rank))


if __name__ == '__main__':
    main()
