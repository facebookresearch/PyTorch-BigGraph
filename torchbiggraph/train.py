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
from itertools import chain, groupby, islice, product
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import attr
import torch
import torch.distributed as td
import torch.multiprocessing as mp
from torch.optim import Optimizer, Adagrad

from .config import parse_config, ConfigSchema
from .entitylist import EntityList
from .eval import eval_many_batches, EvalStats
from .fileio import CheckpointManager, EdgeReader, EntityDataType
from .lockserver import setup_lock_server
from .model import make_model, override_model, MultiRelationEmbedder
from .parameterserver import setup_parameter_server_thread, \
    setup_parameter_server
from .row_adagrad import RowAdagrad
from .util import log, vlog, chunk_by_index, get_partitioned_types, \
    create_pool, fast_approx_rand, DummyOptimizer, create_ordered_buckets, Side, \
    init_process_group, infer_input_index_base, Bucket, Partition, EntityName, \
    Rank, ModuleStateDict, OptimizerStateDict, split_almost_equally
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
    B = batch_lhs.size(0)
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
        count=B)

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
        offset, N = 0, rel.size(0)
        while offset < N:
            B = min(N - offset, config.batch_size)
            all_stats.append(train_one_batch(
                model, optimizers,
                lhs[offset:offset + B],
                rhs[offset:offset + B],
                rel[offset:offset + B]))
            offset += B
    else:
        # group the edges by relation, and only do batches of a single
        # relation type
        _, lhs_chunked, rhs_chunked = chunk_by_index(rel, lhs, rhs)
        edge_count_by_relation = torch.LongTensor([e.nelement() for e in lhs_chunked])
        offset_by_relation = torch.LongTensor([0 for e in lhs_chunked])

        while edge_count_by_relation.sum() > 0:
            # pick which relation to do proportional to number of edges of that type
            batch_rel = torch.multinomial(edge_count_by_relation.float(), 1).item()
            B = min(edge_count_by_relation[batch_rel].item(), config.batch_size)
            offset = offset_by_relation[batch_rel]
            lhs_rel = lhs_chunked[batch_rel][offset:offset + B]
            rhs_rel = rhs_chunked[batch_rel][offset:offset + B]

            all_stats.append(train_one_batch(
                model, optimizers, lhs_rel, rhs_rel, batch_rel))

            edge_count_by_relation[batch_rel] -= B
            offset_by_relation[batch_rel] += B

    return TrainStats.sum(all_stats)


def perform_action_one_thread(
    rank: Rank,
    config: ConfigSchema,
    action: Action,
    model: MultiRelationEmbedder,
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
        if rank > 0:
            time.sleep(config.hogwild_delay)
        stats = train_many_batches(config, model, optimizers, lhs, rhs, rel)
    elif action is Action.EVAL:
        with override_model(model,
                            num_uniform_negs=config.eval_num_uniform_negs,
                            num_batch_negs=config.eval_num_batch_negs):
            stats = eval_many_batches(config, model, lhs, rhs, rel)
    else:
        raise NotImplementedError("Unknown action: %s" % action)
    assert stats.count == my_edges.size(0)
    return stats


def distribute_action_among_workers(
    pool: mp.Pool,
    config: ConfigSchema,
    action: Action,
    model: MultiRelationEmbedder,
    lhs: EntityList,
    rhs: EntityList,
    rel: torch.LongTensor,
    edge_perm: torch.LongTensor,
    optimizers: Optional[List[Optimizer]] = None
) -> Union[TrainStats, EvalStats]:
    all_stats = pool.starmap(perform_action_one_thread, [
        (Rank(i), config, action, model, lhs, rhs, rel, edge_perm[s], optimizers)
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
    N: int,
    D: int,
    scale: float,
) -> Tuple[torch.FloatTensor, None]:
    """Initialize embeddings of size N x D.
    """
    # FIXME: Use multi-threaded instead of fast_approx_rand
    vlog("Initializing %s" % entity)
    return fast_approx_rand(N * D).view(N, D).mul_(scale), None


RANK_ZERO = Rank(0)


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

    index_base = infer_input_index_base(config)

    log("Loading entity counts...")
    entity_counts: Dict[str, List[int]] = {}
    for entity, econf in config.entities.items():
        entity_counts[entity] = []
        for part in range(econf.num_partitions):
            name = "entity_count_%s_%d.pt" % (entity, part + index_base)
            path = os.path.join(config.entity_path, name)
            entity_counts[entity].append(torch.load(path))

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
            world_size=world_size,
            num_clients=config.num_machines,
            init_method=init_method,
            groups=[barrier_group_ranks])

        log("Setup param server...")

        parameter_client = setup_parameter_server_thread(
            client_rank=config.num_machines * 2 + rank,
            server_rank=config.num_machines + rank,
            all_server_ranks=[config.num_machines + x
                              for x in range(config.num_machines)],
            num_clients=config.num_machines,
            world_size=world_size,
            init_method=init_method,
            groups=[barrier_group_ranks])

        num_partition_servers = config.num_partition_servers
        if config.num_partition_servers == -1:
            setup_parameter_server(server_rank=config.num_machines * 3 + 1 + rank,
                                   num_clients=config.num_machines,
                                   world_size=world_size,
                                   init_method=init_method,
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

    def make_optimizer(params, is_emb):
        params = list(params)
        if len(params) == 0:
            return DummyOptimizer()
        constructor = RowAdagrad if is_emb else Adagrad
        optimizer = constructor(params, lr=config.lr)
        optimizer.share_memory()
        return optimizer

    # background_io is only supported in single-machine mode
    background_io = config.background_io and config.num_machines == 1

    checkpoint_manager = CheckpointManager(config.checkpoint_path,
            background=background_io,
            rank=rank,
            num_machines=config.num_machines,
            partition_server_ranks=partition_server_ranks,
            index_base=index_base)
    if config.init_path is not None:
        loadpath_manager = CheckpointManager(config.init_path)
    else:
        loadpath_manager = None

    def load_embeddings(
        entity: EntityName,
        part: Partition,
        strict: bool = False,
        force_dirty: bool = False,
    ) -> EntityDataType:
        data = checkpoint_manager.read(entity, part, strict=strict,
                                       force_dirty=force_dirty)
        if data is None and loadpath_manager is not None:
            data = loadpath_manager.read(entity, part)
        if data is None:
            data = init_embs(entity, entity_counts[entity][part],
                             config.dimension, config.init_scale)
        embs, optim_state = data
        if not isinstance(embs, torch.nn.Parameter):  # Pytorch bug workaround
            embs = torch.nn.Parameter(embs)
        embs.share_memory_()
        return embs, optim_state

    # Figure out how many lhs and rhs partitions we need
    (nparts_lhs, nparts_rhs,
     lhs_partitioned_types, rhs_partitioned_types) = get_partitioned_types(config)
    vlog("nparts %d %d types %s %s" %
         (nparts_lhs, nparts_rhs, lhs_partitioned_types, rhs_partitioned_types))

    log("Initializing global model...")
    assert config.batch_size % config.num_batch_negs == 0, (
        "You really want this to avoid padding every batch")

    model = make_model(config)

    vlog("Initializing optimizers")
    # FIXME: use SGD with different learning rate?
    optimizers: Dict[Optional[Tuple[EntityName, Partition]], Optimizer] = {
        # The "None" optimizer is for the non-entity-specific parameters.
        None: make_optimizer(model.parameters(), False)
    }
    _, edge_path_count, echunk, state_dict, optim_state = \
        checkpoint_manager.read_metadata()
    iteration_idx = edge_path_count * config.num_edge_chunks + echunk
    del edge_path_count
    del echunk
    iterations_left = islice(product(range(config.num_epochs),
                                     range(len(config.edge_paths)),
                                     range(config.num_edge_chunks)),
                             iteration_idx + 1, None)

    if state_dict is None and loadpath_manager is not None:
        _, _, _, state_dict, optim_state = loadpath_manager.read_metadata()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    if optim_state is not None:
        optimizers[None].load_state_dict(optim_state)
        optimizers[None].share_memory()

    model.share_memory()

    vlog("Loading unpartitioned entities...")
    max_parts = max(e.num_partitions for e in config.entities.values())
    for entity, econfig in config.entities.items():
        num_parts = econfig.num_partitions
        assert num_parts == 1 or num_parts == max_parts, (
            "Currently num_partitions must be either 1 or a single value across "
            "all entities.")
        if num_parts == 1:
            embs, optim_state = load_embeddings(entity, Partition(0))
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)
            optimizer = make_optimizer([embs], True)
            if optim_state is not None:
                optimizer.load_state_dict(optim_state)
                optimizer.share_memory()
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
        oldP: Optional[Bucket],
        newP: Optional[Bucket],
    ):
        # 0. given the oldPair and newPair, construct data structures to keep
        #    track of old and new embedding (entity, part) tuples

        io_bytes = 0
        log("Swapping partitioned embeddings %s %s" % (oldP, newP))

        types = ([(e, Side.LHS) for e in lhs_partitioned_types] +
                 [(e, Side.RHS) for e in rhs_partitioned_types])
        old_parts = {(e, oldP.get_partition(side)): side
                     for e, side in types if oldP is not None}
        new_parts = {(e, newP.get_partition(side)): side
                     for e, side in types if newP is not None}

        to_checkpoint = set(old_parts) - set(new_parts)
        preserved = set(old_parts) & set(new_parts)

        # 1. checkpoint embeddings that will not be used in the next pair
        #
        if oldP is not None:  # there are previous embeddings to checkpoint
            log("Writing partitioned embeddings")
            for entity, part in to_checkpoint:
                side = old_parts[(entity, part)]
                vlog("Checkpointing (%s %d %s)" %
                     (entity, part, side.pick("lhs", "rhs")))
                embs = model.get_embeddings(entity, side)
                optim_key = (entity, part)
                optim_state = OptimizerStateDict(optimizers[optim_key].state_dict())
                io_bytes += embs.nelement() * 4  # ignore optim state
                checkpoint_manager.write(entity, (embs, optim_state), part)
                if optim_key in optimizers:
                    del optimizers[optim_key]
                # these variables are holding large objects; let them be freed
                del embs
                del optim_state

            if config.num_machines > 1:
                lock_client.release_bucket(oldP)


        # 2. copy old embeddings that will be used in the next pair
        #    into a temporary dictionary
        #
        tmp_emb = {x: model.get_embeddings(x[0], old_parts[x]) for x in preserved}

        for entity, _ in types:
            model.clear_embeddings(entity)

        if newP is None:  # there are no new embeddings to load
            return io_bytes

        # 3. load new embeddings into the model/optimizer, either from disk
        #    or the temporary dictionary
        #
        log("Loading entities")
        for entity, side in types:
            part = newP.get_partition(side)
            part_key = (entity, part)
            if part_key in tmp_emb:
                vlog("Loading (%s, %d) from preserved" % (entity, part))
                embs, optim_state = tmp_emb[part_key], None
            else:
                vlog("Loading (%s, %d)" % (entity, part))

                force_dirty = (config.num_machines > 1 and
                               lock_client.check_and_set_dirty(entity, part))
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
    for (epoch, edge_path_idx), sub_iterations in groupby(iterations_left, lambda i: i[:2]):
        edge_path = config.edge_paths[edge_path_idx]
        edge_reader = EdgeReader(edge_path, index_base=index_base)
        for _, _, edge_chunk in sub_iterations:
            log("Starting epoch %d / %d edge path %d / %d edge chunk %d / %d" %
                (epoch + 1, config.num_epochs,
                 edge_path_idx + 1, len(config.edge_paths),
                 edge_chunk + 1, config.num_edge_chunks))
            log("edge_path= %s" % edge_path)

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
                                          init_tree=config.distributed_tree_init_order
                                                    and edge_path_idx == 0
                                                    and epoch == 0)
                td.barrier(group=barrier_group)

            # Single-machine case: partition_count keeps track of how
            # many partitions have been processed so far. This is used in
            # pre-fetching only.
            partition_count = 0
            remaining = len(buckets)
            curP = None
            while remaining > 0:
                oldP = curP
                io_time = 0.
                io_bytes = 0
                if config.num_machines > 1:
                    curP, remaining = lock_client.acquire_pair(rank, maybe_old_bucket=oldP)
                    curP = Bucket(*curP) if curP is not None else None
                    print('still in queue: %d' % remaining, file=sys.stderr)
                    if curP is None:
                        if oldP is not None:
                            # if you couldn't get a new pair, release the lock
                            # to prevent a deadlock!
                            tic = time.time()
                            io_bytes += swap_partitioned_embeddings(oldP, None)
                            io_time += time.time() - tic
                        time.sleep(1)  # don't hammer td
                        continue
                else:
                    curP = buckets.pop(0)
                    remaining -= 1

                def log_status(msg, always=False):
                    F = log if always else vlog
                    F("%s: %s" % (curP, msg))

                tic = time.time()

                io_bytes += swap_partitioned_embeddings(oldP, curP)

                current_index = (
                    ((epoch * len(config.edge_paths) + edge_path_idx)
                     * config.num_edge_chunks + edge_chunk + 1)
                    * total_buckets
                    - remaining
                )

                if partition_count < total_buckets - 1 and background_io:
                    assert config.num_machines == 1
                    # Ensure the previous bucket finished writing to disk.
                    checkpoint_manager.wait_for_marker(current_index - 1)

                    log_status("Prefetching")
                    next_partition = buckets[partition_count + 1]
                    for entity in lhs_partitioned_types:
                        checkpoint_manager.prefetch(entity, next_partition.lhs)
                    for entity in rhs_partitioned_types:
                        checkpoint_manager.prefetch(entity, next_partition.rhs)

                    checkpoint_manager.record_marker(current_index)

                log_status("Loading edges")
                lhs, rhs, rel = edge_reader.read(
                    curP.lhs, curP.rhs, edge_chunk, config.num_edge_chunks)
                N = rel.size(0)
                # this might be off in the case of tensorlist
                io_bytes += (lhs.nelement() + rhs.nelement() + rel.nelement()) * 4

                log_status("Shuffling edges")
                # Fix a seed to get the same permutation every time; have it
                # depend on all and only what affects the set of edges.
                g = torch.Generator()
                g.manual_seed(hash((edge_path_idx, edge_chunk, curP.lhs, curP.rhs)))

                num_eval_edges = int(N * config.eval_fraction)
                if num_eval_edges > 0:
                    edge_perm = torch.randperm(N, generator=g)
                    eval_edge_perm = edge_perm[-num_eval_edges:]
                    N -= num_eval_edges
                    edge_perm = edge_perm[torch.randperm(N)]
                else:
                    edge_perm = torch.randperm(N)

                # HOGWILD evaluation before training
                eval_stats_before: Optional[EvalStats] = None
                if num_eval_edges > 0:
                    log_status("Waiting for workers to perform evaluation")
                    eval_stats_before = distribute_action_among_workers(
                        pool, config,
                        Action.EVAL, model, lhs, rhs, rel, eval_edge_perm)
                    log("stats before %s: %s" % (curP, eval_stats_before))

                io_time += time.time() - tic
                tic = time.time()
                # HOGWILD training
                log_status("Waiting for workers to perform training")
                stats = distribute_action_among_workers(
                    pool, config,
                    Action.TRAIN, model, lhs, rhs, rel, edge_perm,
                    list(optimizers.values()))
                compute_time = time.time() - tic

                log_status("bucket %d / %d : Processed %d edges in %.2f s "
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
                        Action.EVAL, model, lhs, rhs, rel, eval_edge_perm)
                    log("stats after %s: %s" % (curP, eval_stats_after))

                # Add train/eval metrics to queue
                yield current_index, eval_stats_before, stats, eval_stats_after

                partition_count += 1

            swap_partitioned_embeddings(curP, None)

            # Distributed Processing: all machines can leave the barrier now.
            if config.num_machines > 1:
                td.barrier(barrier_group)

            # Write metadata: for multiple machines, write from rank-0
            log("Finished epoch %d path %d pass %d; checkpointing global state."
                % (epoch + 1, edge_path_idx + 1, edge_chunk + 1))
            log("My rank: %d" % rank)
            if rank == 0:
                for entity, econfig in config.entities.items():
                    if econfig.num_partitions == 1:
                        embs = model.get_embeddings(entity, Side.LHS)
                        optimizer = optimizers[(entity, Partition(0))]

                        checkpoint_manager.write(
                            entity,
                            (embs, OptimizerStateDict(optimizer.state_dict())),
                            Partition(0)
                        )

                sanitized_state_dict: ModuleStateDict = {}
                for k, v in ModuleStateDict(model.state_dict()).items():
                    if k.startswith('lhs_embs') or k.startswith('rhs_embs'):
                        # skipping state that's an entity embedding
                        continue
                    sanitized_state_dict[k] = v

                log("Writing metadata...")
                checkpoint_manager.write_metadata(
                    config,
                    epoch * len(config.edge_paths) + edge_path_idx,
                    edge_chunk,
                    sanitized_state_dict,
                    OptimizerStateDict(optimizers[None].state_dict()),
                )

            log("Committing checkpoints...")
            checkpoint_manager.commit(config)

            if config.num_machines > 1:
                log("Waiting on barrier: rank %d" % rank)
                td.barrier(barrier_group)
                log("Done barrier")

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
