#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
import math
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as td
from torch.optim import Optimizer
from torchbiggraph.async_adagrad import AsyncAdagrad
from torchbiggraph.batching import AbstractBatchProcessor, call, process_in_batches
from torchbiggraph.bucket_scheduling import (
    BucketStats,
    DistributedBucketScheduler,
    LockServer,
    SingleMachineBucketScheduler,
)
from torchbiggraph.checkpoint_manager import (
    CheckpointManager,
    ConfigMetadataProvider,
    MetadataProvider,
    PartitionClient,
)
from torchbiggraph.config import ConfigSchema
from torchbiggraph.distributed import ProcessRanks, init_process_group, start_server
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.losses import LOSS_FUNCTIONS, AbstractLossFunction
from torchbiggraph.model import MultiRelationEmbedder, make_model
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.types import (
    SINGLE_TRAINER,
    UNPARTITIONED,
    Bucket,
    EntityName,
    FloatTensorType,
    ModuleStateDict,
    Partition,
    Rank,
)
from torchbiggraph.util import (
    BucketLogger,
    DummyOptimizer,
    EmbeddingHolder,
    allocate_shared_tensor,
    create_pool,
    fast_approx_rand,
    get_async_result,
    get_num_workers,
    hide_distributed_logging,
    round_up_to_nearest_multiple,
    split_almost_equally,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


class Trainer(AbstractBatchProcessor):
    def __init__(
        self,
        model_optimizer: Optimizer,
        loss_fn: AbstractLossFunction,
        relation_weights: List[float],
    ) -> None:
        super().__init__(loss_fn, relation_weights)
        self.model_optimizer = model_optimizer
        self.unpartitioned_optimizers: Dict[EntityName, Optimizer] = {}
        self.partitioned_optimizers: Dict[Tuple[EntityName, Partition], Optimizer] = {}

    def _process_one_batch(
        self, model: MultiRelationEmbedder, batch_edges: EdgeList
    ) -> Stats:
        model.zero_grad()

        scores, reg = model(batch_edges)

        loss = self.calc_loss(scores, batch_edges)

        stats = Stats(
            loss=float(loss),
            reg=float(reg) if reg is not None else 0.0,
            violators_lhs=int((scores.lhs_neg > scores.lhs_pos.unsqueeze(1)).sum()),
            violators_rhs=int((scores.rhs_neg > scores.rhs_pos.unsqueeze(1)).sum()),
            count=len(batch_edges),
        )
        if reg is not None:
            (loss + reg).backward()
        else:
            loss.backward()
        self.model_optimizer.step(closure=None)
        for optimizer in self.unpartitioned_optimizers.values():
            optimizer.step(closure=None)
        for optimizer in self.partitioned_optimizers.values():
            optimizer.step(closure=None)

        return stats


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

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
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

    def __add__(self, delta: int) -> "IterationManager":
        return IterationManager(
            self.num_epochs,
            self.edge_paths,
            self.num_edge_chunks,
            iteration_idx=self.iteration_idx + delta,
        )


def should_preserve_old_checkpoint(
    iteration_manager: IterationManager, interval: Optional[int]
) -> bool:
    """Whether the checkpoint consumed by the current iteration should be kept

    Given the period, in number of epochs, at which to snapshot checkpoints,
    determinen whether the checkpoint that is used as input by the current
    iteration (as determined by the given manager) should be preserved rather
    than getting cleaned up.
    """
    if interval is None:
        return False
    is_checkpoint_epoch = iteration_manager.epoch_idx % interval == 0
    is_first_edge_path = iteration_manager.edge_path_idx == 0
    is_first_edge_chunk = iteration_manager.edge_chunk_idx == 0
    return is_checkpoint_epoch and is_first_edge_path and is_first_edge_chunk


def get_num_edge_chunks(config: ConfigSchema) -> int:
    if config.num_edge_chunks is not None:
        return config.num_edge_chunks

    max_edges_per_bucket = 0
    # We should check all edge paths, all lhs partitions and all rhs partitions,
    # but the combinatorial explosion could lead to thousands of checks. Let's
    # assume that edges are uniformly distributed among buckets (this is not
    # exactly the case, as it's the entities that are uniformly distributed
    # among the partitions, and edge assignments to buckets are a function of
    # that, thus, for example, very high degree entities could skew this), and
    # use the size of bucket (0, 0) as an estimate of the average bucket size.
    # We still do it for all edge paths as there could be semantic differences
    # between them which lead to different sizes.
    for edge_path in config.edge_paths:
        edge_storage = EDGE_STORAGES.make_instance(edge_path)
        max_edges_per_bucket = max(
            max_edges_per_bucket,
            edge_storage.get_number_of_edges(UNPARTITIONED, UNPARTITIONED),
        )
    return max(1, math.ceil(max_edges_per_bucket / config.max_edges_per_chunk))


def make_optimizer(
    config: ConfigSchema, params: Iterable[torch.nn.Parameter], is_emb: bool
) -> Optimizer:
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
        optimizer = AsyncAdagrad(params, lr=lr)
    optimizer.share_memory()
    return optimizer


NOOP_STATS_HANDLER = StatsHandler()


class TrainingCoordinator:
    def __init__(  # noqa
        self,
        config: ConfigSchema,
        model: Optional[MultiRelationEmbedder] = None,
        trainer: Optional[AbstractBatchProcessor] = None,
        evaluator: Optional[AbstractBatchProcessor] = None,
        rank: Rank = SINGLE_TRAINER,
        subprocess_init: Optional[Callable[[], None]] = None,
        stats_handler: StatsHandler = NOOP_STATS_HANDLER,
    ):
        """Each epoch/pass, for each partition pair, loads in embeddings and edgelist
        from disk, runs HOGWILD training on them, and writes partitions back to disk.
        """
        tag_logs_with_process_name(f"Trainer-{rank}")
        self.config = config
        if config.verbose > 0:
            import pprint

            pprint.PrettyPrinter().pprint(config.to_dict())

        logger.info("Loading entity counts...")
        entity_storage = ENTITY_STORAGES.make_instance(config.entity_path)
        entity_counts: Dict[str, List[int]] = {}
        for entity, econf in config.entities.items():
            entity_counts[entity] = []
            for part in range(econf.num_partitions):
                entity_counts[entity].append(entity_storage.load_count(entity, part))

        # Figure out how many lhs and rhs partitions we need
        holder = self.holder = EmbeddingHolder(config)

        logger.debug(
            f"nparts {holder.nparts_lhs} {holder.nparts_rhs} "
            f"types {holder.lhs_partitioned_types} {holder.rhs_partitioned_types}"
        )

        # We know ahead of time that we wil need 1-2 storages for each embedding type,
        # as well as the max size of this storage (num_entities x D).
        # We allocate these storages n advance in `embedding_storage_freelist`.
        # When we need storage for an entity type, we pop it from this free list,
        # and then add it back when we 'delete' the embedding table.
        embedding_storage_freelist: Dict[
            EntityName, Set[torch.FloatStorage]
        ] = defaultdict(set)
        for entity_type, counts in entity_counts.items():
            max_count = max(counts)
            num_sides = (
                (1 if entity_type in holder.lhs_partitioned_types else 0)
                + (1 if entity_type in holder.rhs_partitioned_types else 0)
                + (
                    1
                    if entity_type
                    in (holder.lhs_unpartitioned_types | holder.rhs_unpartitioned_types)
                    else 0
                )
            )
            for _ in range(num_sides):
                embedding_storage_freelist[entity_type].add(
                    allocate_shared_tensor(
                        (max_count, config.entity_dimension(entity_type)),
                        dtype=torch.float,
                    ).storage()
                )

        # create the handlers, threads, etc. for distributed training
        if config.num_machines > 1 or config.num_partition_servers > 0:
            if not 0 <= rank < config.num_machines:
                raise RuntimeError("Invalid rank for trainer")
            if not td.is_available():
                raise RuntimeError(
                    "The installed PyTorch version doesn't provide "
                    "distributed training capabilities."
                )
            ranks = ProcessRanks.from_num_invocations(
                config.num_machines, config.num_partition_servers
            )

            num_ps_groups = config.num_groups_for_partition_server
            groups: List[List[int]] = [ranks.trainers]  # barrier group
            groups += [
                ranks.trainers + ranks.partition_servers
            ] * num_ps_groups  # ps groups
            group_idxs_for_partition_servers = range(1, len(groups))

            if rank == SINGLE_TRAINER:
                logger.info("Setup lock server...")
                start_server(
                    LockServer(
                        num_clients=len(ranks.trainers),
                        nparts_lhs=holder.nparts_lhs,
                        nparts_rhs=holder.nparts_rhs,
                        entities_lhs=holder.lhs_partitioned_types,
                        entities_rhs=holder.rhs_partitioned_types,
                        entity_counts=entity_counts,
                        init_tree=config.distributed_tree_init_order,
                        stats_handler=stats_handler,
                    ),
                    process_name="LockServer",
                    init_method=config.distributed_init_method,
                    world_size=ranks.world_size,
                    server_rank=ranks.lock_server,
                    groups=groups,
                    subprocess_init=subprocess_init,
                )

            self.bucket_scheduler = DistributedBucketScheduler(
                server_rank=ranks.lock_server, client_rank=ranks.trainers[rank]
            )

            logger.info("Setup param server...")
            start_server(
                ParameterServer(num_clients=len(ranks.trainers)),
                process_name=f"ParamS-{rank}",
                init_method=config.distributed_init_method,
                world_size=ranks.world_size,
                server_rank=ranks.parameter_servers[rank],
                groups=groups,
                subprocess_init=subprocess_init,
            )

            parameter_sharer = ParameterSharer(
                process_name=f"ParamC-{rank}",
                client_rank=ranks.parameter_clients[rank],
                all_server_ranks=ranks.parameter_servers,
                init_method=config.distributed_init_method,
                world_size=ranks.world_size,
                groups=groups,
                subprocess_init=subprocess_init,
            )

            if config.num_partition_servers == -1:
                start_server(
                    ParameterServer(
                        num_clients=len(ranks.trainers),
                        group_idxs=group_idxs_for_partition_servers,
                        log_stats=True,
                    ),
                    process_name=f"PartS-{rank}",
                    init_method=config.distributed_init_method,
                    world_size=ranks.world_size,
                    server_rank=ranks.partition_servers[rank],
                    groups=groups,
                    subprocess_init=subprocess_init,
                )

            groups = init_process_group(
                rank=ranks.trainers[rank],
                world_size=ranks.world_size,
                init_method=config.distributed_init_method,
                groups=groups,
            )
            trainer_group, *groups_for_partition_servers = groups
            self.barrier_group = trainer_group

            if len(ranks.partition_servers) > 0:
                partition_client = PartitionClient(
                    ranks.partition_servers,
                    groups=groups_for_partition_servers,
                    log_stats=True,
                )
            else:
                partition_client = None
        else:
            self.barrier_group = None
            self.bucket_scheduler = SingleMachineBucketScheduler(
                holder.nparts_lhs, holder.nparts_rhs, config.bucket_order, stats_handler
            )
            parameter_sharer = None
            partition_client = None
            hide_distributed_logging()

        # fork early for HOGWILD threads
        logger.info("Creating workers...")
        self.num_workers = get_num_workers(config.workers)
        self.pool = create_pool(
            self.num_workers,
            subprocess_name=f"TWorker-{rank}",
            subprocess_init=subprocess_init,
        )

        checkpoint_manager = CheckpointManager(
            config.checkpoint_path,
            rank=rank,
            num_machines=config.num_machines,
            partition_client=partition_client,
            subprocess_name=f"BackgRW-{rank}",
            subprocess_init=subprocess_init,
        )
        self.checkpoint_manager = checkpoint_manager
        checkpoint_manager.register_metadata_provider(ConfigMetadataProvider(config))
        if rank == 0:
            checkpoint_manager.write_config(config)

        num_edge_chunks = get_num_edge_chunks(config)

        self.iteration_manager = IterationManager(
            config.num_epochs,
            config.edge_paths,
            num_edge_chunks,
            iteration_idx=checkpoint_manager.checkpoint_version,
        )
        checkpoint_manager.register_metadata_provider(self.iteration_manager)

        logger.info("Initializing global model...")
        if model is None:
            model = make_model(config)
        model.share_memory()
        loss_fn = LOSS_FUNCTIONS.get_class(config.loss_fn)(margin=config.margin)
        relation_weights = [relation.weight for relation in config.relations]
        if trainer is None:
            trainer = Trainer(
                model_optimizer=make_optimizer(config, model.parameters(), False),
                loss_fn=loss_fn,
                relation_weights=relation_weights,
            )
        if evaluator is None:
            eval_overrides = {}
            if config.eval_num_batch_negs is not None:
                eval_overrides["num_batch_negs"] = config.eval_num_batch_negs
            if config.eval_num_uniform_negs is not None:
                eval_overrides["num_uniform_negs"] = config.eval_num_uniform_negs

            evaluator = RankingEvaluator(
                loss_fn=loss_fn,
                relation_weights=relation_weights,
                overrides=eval_overrides,
            )

        if config.init_path is not None:
            self.loadpath_manager = CheckpointManager(config.init_path)
        else:
            self.loadpath_manager = None

        # load model from checkpoint or loadpath, if available
        state_dict, optim_state = checkpoint_manager.maybe_read_model()
        if state_dict is None and self.loadpath_manager is not None:
            state_dict, optim_state = self.loadpath_manager.maybe_read_model()
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        if optim_state is not None:
            trainer.model_optimizer.load_state_dict(optim_state)

        logger.debug("Loading unpartitioned entities...")
        for entity in holder.lhs_unpartitioned_types | holder.rhs_unpartitioned_types:
            count = entity_counts[entity][0]
            s = embedding_storage_freelist[entity].pop()
            dimension = config.entity_dimension(entity)
            embs = torch.FloatTensor(s).view(-1, dimension)[:count]
            embs, optimizer = self._load_embeddings(entity, UNPARTITIONED, out=embs)
            holder.unpartitioned_embeddings[entity] = embs
            trainer.unpartitioned_optimizers[entity] = optimizer

        # start communicating shared parameters with the parameter server
        if parameter_sharer is not None:
            shared_parameters: Set[int] = set()
            for name, param in model.named_parameters():
                if id(param) in shared_parameters:
                    continue
                shared_parameters.add(id(param))
                key = f"model.{name}"
                logger.info(
                    f"Adding {key} ({param.numel()} params) to parameter server"
                )
                parameter_sharer.set_param(key, param.data)
            for entity, embs in holder.unpartitioned_embeddings.items():
                key = f"entity.{entity}"
                logger.info(f"Adding {key} ({embs.numel()} params) to parameter server")
                parameter_sharer.set_param(key, embs.data)

        # store everything in self
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.rank = rank
        self.entity_counts = entity_counts
        self.embedding_storage_freelist = embedding_storage_freelist
        self.stats_handler = stats_handler

        self.strict = False

    def train(self) -> None:

        holder = self.holder
        config = self.config
        iteration_manager = self.iteration_manager

        total_buckets = holder.nparts_lhs * holder.nparts_rhs

        # yield stats from checkpoint, to reconstruct
        # saved part of the learning curve
        if self.rank == SINGLE_TRAINER:
            for stats_dict in self.checkpoint_manager.maybe_read_stats():
                index: int = stats_dict["index"]
                stats: Optional[Stats] = None
                if "stats" in stats_dict:
                    stats: Stats = Stats.from_dict(stats_dict["stats"])
                eval_stats_before: Optional[Stats] = None
                if "eval_stats_before" in stats_dict:
                    eval_stats_before = Stats.from_dict(stats_dict["eval_stats_before"])
                eval_stats_after: Optional[Stats] = None
                if "eval_stats_after" in stats_dict:
                    eval_stats_after = Stats.from_dict(stats_dict["eval_stats_after"])
                eval_stats_chunk_avg: Optional[Stats] = None
                if "eval_stats_chunk_avg" in stats_dict:
                    eval_stats_chunk_avg = Stats.from_dict(
                        stats_dict["eval_stats_chunk_avg"]
                    )
                self.stats_handler.on_stats(
                    index,
                    eval_stats_before,
                    stats,
                    eval_stats_after,
                    eval_stats_chunk_avg,
                )

        for epoch_idx, edge_path_idx, edge_chunk_idx in iteration_manager:
            logger.info(
                f"Starting epoch {epoch_idx + 1} / {iteration_manager.num_epochs}, "
                f"edge path {edge_path_idx + 1} / {iteration_manager.num_edge_paths}, "
                f"edge chunk {edge_chunk_idx + 1} / {iteration_manager.num_edge_chunks}"
            )
            edge_storage = EDGE_STORAGES.make_instance(iteration_manager.edge_path)
            logger.info(f"Edge path: {iteration_manager.edge_path}")

            self._barrier()
            dist_logger.info("Lock client new epoch...")
            self.bucket_scheduler.new_pass(
                is_first=iteration_manager.iteration_idx == 0
            )
            self._barrier()

            remaining = total_buckets
            cur_b: Optional[Bucket] = None
            cur_stats: Optional[BucketStats] = None
            while remaining > 0:
                old_b: Optional[Bucket] = cur_b
                old_stats: Optional[BucketStats] = cur_stats
                cur_b, remaining = self.bucket_scheduler.acquire_bucket()
                logger.info(f"still in queue: {remaining}")
                if cur_b is None:
                    cur_stats = None
                    if old_b is not None:
                        # if you couldn't get a new pair, release the lock
                        # to prevent a deadlock!
                        tic = time.perf_counter()
                        release_bytes = self._swap_partitioned_embeddings(
                            old_b, None, old_stats
                        )
                        release_time = time.perf_counter() - tic
                        logger.info(
                            f"Swapping old embeddings to release lock. io: {release_time:.2f} s for {release_bytes:,} bytes "
                            f"( {release_bytes / release_time / 1e6:.2f} MB/sec )"
                        )
                    time.sleep(1)  # don't hammer td
                    continue

                tic = time.perf_counter()
                self.cur_b = cur_b
                bucket_logger = BucketLogger(logger, bucket=cur_b)
                self.bucket_logger = bucket_logger

                io_bytes = self._swap_partitioned_embeddings(old_b, cur_b, old_stats)
                self.model.set_all_embeddings(holder, cur_b)

                current_index = (
                    iteration_manager.iteration_idx + 1
                ) * total_buckets - remaining

                bucket_logger.debug("Loading edges")
                edges = edge_storage.load_chunk_of_edges(
                    cur_b.lhs,
                    cur_b.rhs,
                    edge_chunk_idx,
                    iteration_manager.num_edge_chunks,
                    shared=True,
                )
                num_edges = len(edges)

                # this might be off in the case of tensorlist or extra edge fields
                io_bytes += edges.lhs.tensor.numel() * edges.lhs.tensor.element_size()
                io_bytes += edges.rhs.tensor.numel() * edges.rhs.tensor.element_size()
                io_bytes += edges.rel.numel() * edges.rel.element_size()
                io_time = time.perf_counter() - tic
                tic = time.perf_counter()
                bucket_logger.debug("Shuffling edges")
                # Fix a seed to get the same permutation every time; have it
                # depend on all and only what affects the set of edges.

                # Note: for the sake of efficiency, we sample eval edge idxs
                # from the edge set *with replacement*, meaning that there may
                # be duplicates of the same edge in the eval set. When we swap
                # edges into the eval set, if there are duplicates then all
                # but one will be clobbered. These collisions are unlikely
                # if eval_fraction is small.
                #
                # Importantly, this eval sampling strategy is theoretically
                # sound:
                # * Training and eval sets are (exactly) disjoint
                # * Eval set may have (rare) duplicates, but they are
                #   uniformly sampled so it's still an unbiased estimator
                #   of the out-of-sample statistics
                num_eval_edges = int(num_edges * config.eval_fraction)
                num_train_edges = num_edges - num_eval_edges
                if num_eval_edges > 0:
                    g = torch.Generator()
                    g.manual_seed(
                        hash((edge_path_idx, edge_chunk_idx, cur_b.lhs, cur_b.rhs))
                    )
                    eval_edge_idxs = torch.randint(
                        num_edges, (num_eval_edges,), dtype=torch.long, generator=g
                    )
                else:
                    eval_edge_idxs = None

                # HOGWILD evaluation before training
                eval_stats_before = self._coordinate_eval(edges, eval_edge_idxs)
                if eval_stats_before is not None:
                    bucket_logger.info(f"Stats before training: {eval_stats_before}")
                eval_time = time.perf_counter() - tic
                tic = time.perf_counter()

                # HOGWILD training
                bucket_logger.debug("Waiting for workers to perform training")
                stats = self._coordinate_train(edges, eval_edge_idxs, epoch_idx)
                if stats is not None:
                    bucket_logger.info(f"Training stats: {stats}")
                train_time = time.perf_counter() - tic
                tic = time.perf_counter()

                # HOGWILD evaluation after training
                eval_stats_after = self._coordinate_eval(edges, eval_edge_idxs)
                if eval_stats_after is not None:
                    bucket_logger.info(f"Stats after training: {eval_stats_after}")

                eval_time += time.perf_counter() - tic

                bucket_logger.info(
                    f"bucket {total_buckets - remaining} / {total_buckets} : "
                    f"Trained {num_train_edges} edges in {train_time:.2f} s "
                    f"( {num_train_edges / train_time / 1e6:.2g} M/sec ); "
                    f"Eval 2*{num_eval_edges} edges in {eval_time:.2f} s "
                    f"( {2 * num_eval_edges / eval_time / 1e6:.2g} M/sec ); "
                    f"io: {io_time:.2f} s for {io_bytes:,} bytes ( {io_bytes / io_time / 1e6:.2f} MB/sec )"
                )

                self.model.clear_all_embeddings()

                cur_stats = BucketStats(
                    lhs_partition=cur_b.lhs,
                    rhs_partition=cur_b.rhs,
                    index=current_index,
                    train=stats,
                    eval_before=eval_stats_before,
                    eval_after=eval_stats_after,
                )

            # release the final bucket
            self._swap_partitioned_embeddings(cur_b, None, cur_stats)

            # Distributed Processing: all machines can leave the barrier now.
            self._barrier()

            current_index = (iteration_manager.iteration_idx + 1) * total_buckets - 1

            self._maybe_write_checkpoint(
                epoch_idx, edge_path_idx, edge_chunk_idx, current_index
            )

            # now we're sure that all partition files exist,
            # so be strict about loading them
            self.strict = True

    def close(self):
        # cleanup
        self.pool.close()
        self.pool.join()

        self._barrier()

        self.checkpoint_manager.close()
        if self.loadpath_manager is not None:
            self.loadpath_manager.close()

        # FIXME join distributed workers (not really necessary)

        logger.info("Exiting")

    ###########################################################################
    # private functions
    ###########################################################################

    def _barrier(self) -> None:
        if self.barrier_group is not None:
            td.barrier(group=self.barrier_group)

    def _load_embeddings(
        self,
        entity: EntityName,
        part: Partition,
        out: FloatTensorType,
        strict: bool = False,
        force_dirty: bool = False,
    ) -> Tuple[torch.nn.Parameter, Optimizer]:
        if strict:
            embs, optim_state = self.checkpoint_manager.read(
                entity, part, out=out, force_dirty=force_dirty
            )
        else:
            # Strict is only false during the first iteration, because in that
            # case the checkpoint may not contain any data (unless a previous
            # run was resumed) so we fall back on initial values.
            embs, optim_state = self.checkpoint_manager.maybe_read(
                entity, part, out=out, force_dirty=force_dirty
            )
            if embs is None and self.loadpath_manager is not None:
                embs, optim_state = self.loadpath_manager.maybe_read(
                    entity, part, out=out
                )
            if embs is None:
                embs = out
                fast_approx_rand(embs)
                embs.mul_(self.config.init_scale)
                optim_state = None
        embs = torch.nn.Parameter(embs)
        optimizer = make_optimizer(self.config, [embs], True)
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        return embs, optimizer

    def _swap_partitioned_embeddings(
        self,
        old_b: Optional[Bucket],
        new_b: Optional[Bucket],
        old_stats: Optional[BucketStats],
    ) -> int:
        io_bytes = 0
        logger.info(f"Swapping partitioned embeddings {old_b} {new_b}")

        holder = self.holder
        old_parts: Set[Tuple[EntityName, Partition]] = set()
        if old_b is not None:
            old_parts.update((e, old_b.lhs) for e in holder.lhs_partitioned_types)
            old_parts.update((e, old_b.rhs) for e in holder.rhs_partitioned_types)
        new_parts: Set[Tuple[EntityName, Partition]] = set()
        if new_b is not None:
            new_parts.update((e, new_b.lhs) for e in holder.lhs_partitioned_types)
            new_parts.update((e, new_b.rhs) for e in holder.rhs_partitioned_types)

        assert old_parts == holder.partitioned_embeddings.keys()

        if old_b is not None:
            if old_stats is None:
                raise TypeError("Got old bucket but not its stats")
            logger.info("Saving partitioned embeddings to checkpoint")
            for entity, part in old_parts - new_parts:
                logger.debug(f"Saving ({entity} {part})")
                embs = holder.partitioned_embeddings.pop((entity, part))
                optimizer = self.trainer.partitioned_optimizers.pop((entity, part))
                self.checkpoint_manager.write(
                    entity, part, embs.detach(), optimizer.state_dict()
                )
                self.embedding_storage_freelist[entity].add(embs.storage())
                io_bytes += embs.numel() * embs.element_size()  # ignore optim state
                # these variables are holding large objects; let them be freed
                del embs
                del optimizer

            self.bucket_scheduler.release_bucket(old_b, old_stats)

        if new_b is not None:
            logger.info("Loading partitioned embeddings from checkpoint")
            for entity, part in new_parts - old_parts:
                logger.debug(f"Loading ({entity} {part})")
                force_dirty = self.bucket_scheduler.check_and_set_dirty(entity, part)
                count = self.entity_counts[entity][part]
                s = self.embedding_storage_freelist[entity].pop()
                dimension = self.config.entity_dimension(entity)
                embs = torch.FloatTensor(s).view(-1, dimension)[:count]
                embs, optimizer = self._load_embeddings(
                    entity, part, out=embs, strict=self.strict, force_dirty=force_dirty
                )
                holder.partitioned_embeddings[entity, part] = embs
                self.trainer.partitioned_optimizers[entity, part] = optimizer
                io_bytes += embs.numel() * embs.element_size()  # ignore optim state

        assert new_parts == holder.partitioned_embeddings.keys()

        return io_bytes

    def _coordinate_train(self, edges, eval_edge_idxs, epoch_idx) -> Stats:
        assert self.config.num_gpus == 0, "GPU training not supported"

        if eval_edge_idxs is not None:
            num_train_edges = len(edges) - len(eval_edge_idxs)
            train_edge_idxs = torch.arange(len(edges))
            train_edge_idxs[eval_edge_idxs] = torch.arange(num_train_edges, len(edges))
            train_edge_idxs = train_edge_idxs[:num_train_edges]
            edge_perm = train_edge_idxs[torch.randperm(num_train_edges)]
        else:
            edge_perm = torch.randperm(len(edges))

        future_all_stats = self.pool.map_async(
            call,
            [
                partial(
                    process_in_batches,
                    batch_size=self.config.batch_size,
                    model=self.model,
                    batch_processor=self.trainer,
                    edges=edges,
                    indices=edge_perm[s],
                    # FIXME should we only delay if iteration_idx == 0?
                    delay=self.config.hogwild_delay
                    if epoch_idx == 0 and self.rank > 0
                    else 0,
                )
                for rank, s in enumerate(
                    split_almost_equally(edge_perm.size(0), num_parts=self.num_workers)
                )
            ],
        )
        all_stats = get_async_result(future_all_stats, self.pool)
        return Stats.sum(all_stats).average()

    def _coordinate_eval(self, edges, eval_edge_idxs) -> Optional[Stats]:
        eval_batch_size = round_up_to_nearest_multiple(
            self.config.batch_size, self.config.eval_num_batch_negs
        )
        if eval_edge_idxs is not None:
            self.bucket_logger.debug("Waiting for workers to perform evaluation")
            future_all_eval_stats = self.pool.map_async(
                call,
                [
                    partial(
                        process_in_batches,
                        batch_size=eval_batch_size,
                        model=self.model,
                        batch_processor=self.evaluator,
                        edges=edges,
                        indices=eval_edge_idxs[s],
                    )
                    for s in split_almost_equally(
                        eval_edge_idxs.size(0), num_parts=self.num_workers
                    )
                ],
            )
            all_eval_stats = get_async_result(future_all_eval_stats, self.pool)
            return Stats.sum(all_eval_stats).average()
        else:
            return None

    def _maybe_write_checkpoint(
        self,
        epoch_idx: int,
        edge_path_idx: int,
        edge_chunk_idx: int,
        current_index: int,
    ) -> None:

        config = self.config

        # Preserving a checkpoint requires two steps:
        # - create a snapshot (w/ symlinks) after it's first written;
        # - don't delete it once the following one is written.
        # These two happen in two successive iterations of the main loop: the
        # one just before and the one just after the epoch boundary.
        preserve_old_checkpoint = should_preserve_old_checkpoint(
            self.iteration_manager, config.checkpoint_preservation_interval
        )
        preserve_new_checkpoint = should_preserve_old_checkpoint(
            self.iteration_manager + 1, config.checkpoint_preservation_interval
        )

        # Write metadata: for multiple machines, write from rank-0
        logger.info(
            f"Finished epoch {epoch_idx + 1} / {self.iteration_manager.num_epochs}, "
            f"edge path {edge_path_idx + 1} / {self.iteration_manager.num_edge_paths}, "
            f"edge chunk {edge_chunk_idx + 1} / "
            f"{self.iteration_manager.num_edge_chunks}"
        )
        if self.rank == 0:
            for entity, embs in self.holder.unpartitioned_embeddings.items():
                logger.info(f"Writing {entity} embeddings")
                optimizer = self.trainer.unpartitioned_optimizers[entity]
                self.checkpoint_manager.write(
                    entity,
                    UNPARTITIONED,
                    embs.detach(),
                    optimizer.state_dict(),
                    unpartitioned=True,
                )

            logger.info("Writing the metadata")
            state_dict: ModuleStateDict = self.model.state_dict()
            self.checkpoint_manager.write_model(
                state_dict, self.trainer.model_optimizer.state_dict()
            )

            logger.info("Writing the training stats")
            all_stats_dicts: List[Dict[str, Any]] = []
            bucket_eval_stats_list = []
            chunk_stats_dict = {
                "epoch_idx": epoch_idx,
                "edge_path_idx": edge_path_idx,
                "edge_chunk_idx": edge_chunk_idx,
            }
            for stats in self.bucket_scheduler.get_stats_for_pass():
                stats_dict = {
                    "lhs_partition": stats.lhs_partition,
                    "rhs_partition": stats.rhs_partition,
                    "index": stats.index,
                    "stats": stats.train.to_dict(),
                }
                if stats.eval_before is not None:
                    stats_dict["eval_stats_before"] = stats.eval_before.to_dict()
                    bucket_eval_stats_list.append(stats.eval_before)

                if stats.eval_after is not None:
                    stats_dict["eval_stats_after"] = stats.eval_after.to_dict()

                stats_dict.update(chunk_stats_dict)
                all_stats_dicts.append(stats_dict)

            if len(bucket_eval_stats_list) != 0:
                eval_stats_chunk_avg = Stats.average_list(bucket_eval_stats_list)
                self.stats_handler.on_stats(
                    index=current_index, eval_stats_chunk_avg=eval_stats_chunk_avg
                )
                chunk_stats_dict["index"] = current_index
                chunk_stats_dict[
                    "eval_stats_chunk_avg"
                ] = eval_stats_chunk_avg.to_dict()
                all_stats_dicts.append(chunk_stats_dict)

            self.checkpoint_manager.append_stats(all_stats_dicts)

        logger.info("Writing the checkpoint")
        self.checkpoint_manager.write_new_version(
            config, self.entity_counts, self.embedding_storage_freelist
        )

        dist_logger.info(
            "Waiting for other workers to write their parts of the checkpoint"
        )
        self._barrier()
        dist_logger.info("All parts of the checkpoint have been written")

        logger.info("Switching to the new checkpoint version")
        self.checkpoint_manager.switch_to_new_version()

        dist_logger.info(
            "Waiting for other workers to switch to the new checkpoint version"
        )
        self._barrier()
        dist_logger.info("All workers have switched to the new checkpoint version")

        # After all the machines have finished committing
        # checkpoints, we either remove the old checkpoints
        # or we preserve it
        if preserve_new_checkpoint:
            # Add 1 so the index is a multiple of the interval, it looks nicer.
            self.checkpoint_manager.preserve_current_version(config, epoch_idx + 1)
        if not preserve_old_checkpoint:
            self.checkpoint_manager.remove_old_version(config)
