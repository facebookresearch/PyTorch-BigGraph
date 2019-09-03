#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
import time
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

import torch
import torch.distributed as td
from torch.optim import Adagrad, Optimizer

from torchbiggraph.batching import (
    AbstractBatchProcessor,
    call,
    process_in_batches,
)
from torchbiggraph.bucket_scheduling import (
    AbstractBucketScheduler,
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
from torchbiggraph.config import (
    add_to_sys_path,
    ConfigFileLoader,
    ConfigSchema,
    RelationSchema,
)
from torchbiggraph.distributed import (
    ProcessRanks,
    init_process_group,
    start_server,
)
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.edgelist_reader import EDGELIST_READERS
from torchbiggraph.entity_count_reader import ENTITY_COUNT_READERS
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.losses import AbstractLossFunction, LOSS_FUNCTIONS
from torchbiggraph.model import (
    MultiRelationEmbedder,
    make_model,
    override_model,
)
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats
from torchbiggraph.types import (
    Bucket,
    EntityName,
    FloatTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
    Side,
)
from torchbiggraph.util import (
    BucketLogger,
    DummyOptimizer,
    create_pool,
    fast_approx_rand,
    get_async_result,
    get_num_workers,
    get_partitioned_types,
    hide_distributed_logging,
    round_up_to_nearest_multiple,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    SubprocessInitializer,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


class Trainer(AbstractBatchProcessor):

    loss_fn: AbstractLossFunction

    def __init__(
        self,
        global_optimizer: Optimizer,
        loss_fn: str,
        margin: float,
        relations: List[RelationSchema],
    ) -> None:
        super().__init__()
        self.global_optimizer = global_optimizer
        self.entity_optimizers: Dict[Tuple[EntityName, Partition], Optimizer] = {}

        loss_fn_class = LOSS_FUNCTIONS.get_class(loss_fn)
        # TODO This is awful! Can we do better?
        if loss_fn == "ranking":
            self.loss_fn = loss_fn_class(margin)
        else:
            self.loss_fn = loss_fn_class()

        self.relations = relations

    def process_one_batch(
        self,
        model: MultiRelationEmbedder,
        batch_edges: EdgeList,
    ) -> Stats:
        model.zero_grad()

        scores = model(batch_edges)

        lhs_loss = self.loss_fn(scores.lhs_pos, scores.lhs_neg)
        rhs_loss = self.loss_fn(scores.rhs_pos, scores.rhs_neg)
        relation = self.relations[batch_edges.get_relation_type_as_scalar()
                                  if batch_edges.has_scalar_relation_type()
                                  else 0]
        loss = relation.weight * (lhs_loss + rhs_loss)

        stats = Stats(
            loss=float(loss),
            violators_lhs=int((scores.lhs_neg > scores.lhs_pos.unsqueeze(1)).sum()),
            violators_rhs=int((scores.rhs_neg > scores.rhs_pos.unsqueeze(1)).sum()),
            count=len(batch_edges))

        loss.backward()
        self.global_optimizer.step(closure=None)
        for optimizer in self.entity_optimizers.values():
            optimizer.step(closure=None)

        return stats


class TrainingRankingEvaluator(RankingEvaluator):

    def __init__(
        self,
        override_num_batch_negs: int,
        override_num_uniform_negs: int,
    ) -> None:
        super().__init__()
        self.override_num_batch_negs = override_num_batch_negs
        self.override_num_uniform_negs = override_num_uniform_negs

    def process_one_batch(
        self,
        model: MultiRelationEmbedder,
        batch_edges: EdgeList,
    ) -> Stats:
        with override_model(model,
                            num_batch_negs=self.override_num_batch_negs,
                            num_uniform_negs=self.override_num_uniform_negs):
            return super().process_one_batch(model, batch_edges)


def init_embs(
    entity: EntityName,
    entity_count: int,
    dim: int,
    scale: float,
) -> Tuple[FloatTensorType, None]:
    """Initialize embeddings of size entity_count x dim.
    """
    # FIXME: Use multi-threaded instead of fast_approx_rand
    logger.debug(f"Initializing {entity}")
    return fast_approx_rand(entity_count * dim).view(entity_count, dim).mul_(scale), None


RANK_ZERO = Rank(0)


class AbstractSynchronizer(ABC):

    @abstractmethod
    def barrier(self) -> None:
        pass


class DummySynchronizer(AbstractSynchronizer):

    def barrier(self):
        pass


class DistributedSynchronizer(AbstractSynchronizer):

    def __init__(self, group: 'td.ProcessGroup') -> None:
        self.group = group

    def barrier(self):
        td.barrier(group=self.group)


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
    iteration_manager: IterationManager,
    interval: Optional[int],
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


def train_and_report_stats(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    trainer: Optional[AbstractBatchProcessor] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    rank: Rank = RANK_ZERO,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> Generator[Tuple[int, Optional[Stats], Stats, Optional[Stats]], None, None]:
    """Each epoch/pass, for each partition pair, loads in embeddings and edgelist
    from disk, runs HOGWILD training on them, and writes partitions back to disk.
    """
    tag_logs_with_process_name(f"Trainer-{rank}")

    if config.verbose > 0:
        import pprint
        pprint.PrettyPrinter().pprint(config.to_dict())

    logger.info("Loading entity counts...")
    entity_count_reader = ENTITY_COUNT_READERS.make_instance(config.entity_path)
    entity_counts: Dict[str, List[int]] = {}
    for entity, econf in config.entities.items():
        entity_counts[entity] = []
        for part in range(econf.num_partitions):
            entity_counts[entity].append(entity_count_reader.read_entity_count(entity, part))

    # Figure out how many lhs and rhs partitions we need
    nparts_lhs, lhs_partitioned_types = get_partitioned_types(config, Side.LHS)
    nparts_rhs, rhs_partitioned_types = get_partitioned_types(config, Side.RHS)
    logger.debug(
        f"nparts {nparts_lhs} {nparts_rhs} "
        f"types {lhs_partitioned_types} {rhs_partitioned_types}")
    total_buckets = nparts_lhs * nparts_rhs

    sync: AbstractSynchronizer
    bucket_scheduler: AbstractBucketScheduler
    parameter_sharer: Optional[ParameterSharer]
    partition_client: Optional[PartitionClient]
    if config.num_machines > 1:
        if not 0 <= rank < config.num_machines:
            raise RuntimeError("Invalid rank for trainer")
        if not td.is_available():
            raise RuntimeError("The installed PyTorch version doesn't provide "
                               "distributed training capabilities.")
        ranks = ProcessRanks.from_num_invocations(
            config.num_machines, config.num_partition_servers)

        if rank == RANK_ZERO:
            logger.info("Setup lock server...")
            start_server(
                LockServer(
                    num_clients=len(ranks.trainers),
                    nparts_lhs=nparts_lhs,
                    nparts_rhs=nparts_rhs,
                    lock_lhs=len(lhs_partitioned_types) > 0,
                    lock_rhs=len(rhs_partitioned_types) > 0,
                    init_tree=config.distributed_tree_init_order,
                ),
                process_name="LockServer",
                init_method=config.distributed_init_method,
                world_size=ranks.world_size,
                server_rank=ranks.lock_server,
                groups=[ranks.trainers],
                subprocess_init=subprocess_init,
            )

        bucket_scheduler = DistributedBucketScheduler(
            server_rank=ranks.lock_server,
            client_rank=ranks.trainers[rank],
        )

        logger.info("Setup param server...")
        start_server(
            ParameterServer(num_clients=len(ranks.trainers)),
            process_name=f"ParamS-{rank}",
            init_method=config.distributed_init_method,
            world_size=ranks.world_size,
            server_rank=ranks.parameter_servers[rank],
            groups=[ranks.trainers],
            subprocess_init=subprocess_init,
        )

        parameter_sharer = ParameterSharer(
            process_name=f"ParamC-{rank}",
            client_rank=ranks.parameter_clients[rank],
            all_server_ranks=ranks.parameter_servers,
            init_method=config.distributed_init_method,
            world_size=ranks.world_size,
            groups=[ranks.trainers],
            subprocess_init=subprocess_init,
        )

        if config.num_partition_servers == -1:
            start_server(
                ParameterServer(num_clients=len(ranks.trainers), log_stats=True),
                process_name=f"PartS-{rank}",
                init_method=config.distributed_init_method,
                world_size=ranks.world_size,
                server_rank=ranks.partition_servers[rank],
                groups=[ranks.trainers],
                subprocess_init=subprocess_init,
            )

        if len(ranks.partition_servers) > 0:
            partition_client = PartitionClient(ranks.partition_servers, log_stats=True)
        else:
            partition_client = None

        groups = init_process_group(
            rank=ranks.trainers[rank],
            world_size=ranks.world_size,
            init_method=config.distributed_init_method,
            groups=[ranks.trainers],
        )
        trainer_group, = groups
        sync = DistributedSynchronizer(trainer_group)

    else:
        sync = DummySynchronizer()
        bucket_scheduler = SingleMachineBucketScheduler(
            nparts_lhs, nparts_rhs, config.bucket_order)
        parameter_sharer = None
        partition_client = None
        hide_distributed_logging()

    # fork early for HOGWILD threads
    logger.info("Creating workers...")
    num_workers = get_num_workers(config.workers)
    pool = create_pool(
        num_workers,
        subprocess_name=f"TWorker-{rank}",
        subprocess_init=subprocess_init,
    )

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
        partition_client=partition_client,
        subprocess_name=f"BackgRW-{rank}",
        subprocess_init=subprocess_init,
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

    logger.info("Initializing global model...")

    if model is None:
        model = make_model(config)
    model.share_memory()
    if trainer is None:
        trainer = Trainer(
            global_optimizer=make_optimizer(model.parameters(), False),
            loss_fn=config.loss_fn,
            margin=config.margin,
            relations=config.relations,
        )
    if evaluator is None:
        evaluator = TrainingRankingEvaluator(
            override_num_batch_negs=config.eval_num_batch_negs,
            override_num_uniform_negs=config.eval_num_uniform_negs,
        )
    eval_batch_size = round_up_to_nearest_multiple(config.batch_size, config.eval_num_batch_negs)

    state_dict, optim_state = checkpoint_manager.maybe_read_model()

    if state_dict is None and loadpath_manager is not None:
        state_dict, optim_state = loadpath_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    if optim_state is not None:
        trainer.global_optimizer.load_state_dict(optim_state)

    logger.debug("Loading unpartitioned entities...")
    for entity, econfig in config.entities.items():
        if econfig.num_partitions == 1:
            embs, optim_state = load_embeddings(entity, Partition(0))
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)
            optimizer = make_optimizer([embs], True)
            if optim_state is not None:
                optimizer.load_state_dict(optim_state)
            trainer.entity_optimizers[(entity, Partition(0))] = optimizer

    # start communicating shared parameters with the parameter server
    if parameter_sharer is not None:
        parameter_sharer.share_model_params(model)

    strict = False

    def swap_partitioned_embeddings(
        old_b: Optional[Bucket],
        new_b: Optional[Bucket],
    ):
        # 0. given the old and new buckets, construct data structures to keep
        #    track of old and new embedding (entity, part) tuples

        io_bytes = 0
        logger.info(f"Swapping partitioned embeddings {old_b} {new_b}")

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
            logger.info("Writing partitioned embeddings")
            for entity, part in to_checkpoint:
                side = old_parts[(entity, part)]
                side_name = side.pick("lhs", "rhs")
                logger.debug(f"Checkpointing ({entity} {part} {side_name})")
                embs = model.get_embeddings(entity, side)
                optim_key = (entity, part)
                optim_state = OptimizerStateDict(trainer.entity_optimizers[optim_key].state_dict())
                io_bytes += embs.numel() * embs.element_size()  # ignore optim state
                checkpoint_manager.write(entity, part, embs.detach(), optim_state)
                if optim_key in trainer.entity_optimizers:
                    del trainer.entity_optimizers[optim_key]
                # these variables are holding large objects; let them be freed
                del embs
                del optim_state

            bucket_scheduler.release_bucket(old_b)

        # 2. copy old embeddings that will be used in the next pair
        #    into a temporary dictionary
        #
        tmp_emb = {x: model.get_embeddings(x[0], old_parts[x]) for x in preserved}

        for entity, _ in types:
            model.clear_embeddings(entity, Side.LHS)
            model.clear_embeddings(entity, Side.RHS)

        if new_b is None:  # there are no new embeddings to load
            return io_bytes

        bucket_logger = BucketLogger(logger, bucket=new_b)

        # 3. load new embeddings into the model/optimizer, either from disk
        #    or the temporary dictionary
        #
        bucket_logger.info("Loading entities")
        for entity, side in types:
            part = new_b.get_partition(side)
            part_key = (entity, part)
            if part_key in tmp_emb:
                bucket_logger.debug(f"Loading ({entity}, {part}) from preserved")
                embs, optim_state = tmp_emb[part_key], None
            else:
                bucket_logger.debug(f"Loading ({entity}, {part})")

                force_dirty = bucket_scheduler.check_and_set_dirty(entity, part)
                embs, optim_state = load_embeddings(
                    entity, part, strict=strict, force_dirty=force_dirty)
                io_bytes += embs.numel() * embs.element_size()  # ignore optim state

            model.set_embeddings(entity, embs, side)
            tmp_emb[part_key] = embs

            optim_key = (entity, part)
            if optim_key not in trainer.entity_optimizers:
                bucket_logger.debug(f"Resetting optimizer {optim_key}")
                optimizer = make_optimizer([embs], True)
                if optim_state is not None:
                    bucket_logger.debug("Setting optim state")
                    optimizer.load_state_dict(optim_state)

                trainer.entity_optimizers[optim_key] = optimizer

        return io_bytes

    # Start of the main training loop.
    for epoch_idx, edge_path_idx, edge_chunk_idx in iteration_manager:
        logger.info(
            f"Starting epoch {epoch_idx + 1} / {iteration_manager.num_epochs}, "
            f"edge path {edge_path_idx + 1} / {iteration_manager.num_edge_paths}, "
            f"edge chunk {edge_chunk_idx + 1} / {iteration_manager.num_edge_chunks}")
        edgelist_reader = EDGELIST_READERS.make_instance(iteration_manager.edge_path)
        logger.info(f"Edge path: {iteration_manager.edge_path}")

        sync.barrier()
        dist_logger.info("Lock client new epoch...")
        bucket_scheduler.new_pass(is_first=iteration_manager.iteration_idx == 0)
        sync.barrier()

        remaining = total_buckets
        cur_b = None
        while remaining > 0:
            old_b = cur_b
            io_time = 0.
            io_bytes = 0
            cur_b, remaining = bucket_scheduler.acquire_bucket()
            logger.info(f"still in queue: {remaining}")
            if cur_b is None:
                if old_b is not None:
                    # if you couldn't get a new pair, release the lock
                    # to prevent a deadlock!
                    tic = time.time()
                    io_bytes += swap_partitioned_embeddings(old_b, None)
                    io_time += time.time() - tic
                time.sleep(1)  # don't hammer td
                continue

            bucket_logger = BucketLogger(logger, bucket=cur_b)

            tic = time.time()

            io_bytes += swap_partitioned_embeddings(old_b, cur_b)

            current_index = \
                (iteration_manager.iteration_idx + 1) * total_buckets - remaining

            next_b = bucket_scheduler.peek()
            if next_b is not None and background_io:
                # Ensure the previous bucket finished writing to disk.
                checkpoint_manager.wait_for_marker(current_index - 1)

                bucket_logger.debug("Prefetching")
                for entity in lhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_b.lhs)
                for entity in rhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_b.rhs)

                checkpoint_manager.record_marker(current_index)

            bucket_logger.debug("Loading edges")
            edges = edgelist_reader.read_edgelist(
                cur_b.lhs, cur_b.rhs, edge_chunk_idx, config.num_edge_chunks)
            num_edges = len(edges)
            # this might be off in the case of tensorlist or extra edge fields
            io_bytes += edges.lhs.tensor.numel() * edges.lhs.tensor.element_size()
            io_bytes += edges.rhs.tensor.numel() * edges.rhs.tensor.element_size()
            io_bytes += edges.rel.numel() * edges.rel.element_size()

            bucket_logger.debug("Shuffling edges")
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
            eval_stats_before: Optional[Stats] = None
            if num_eval_edges > 0:
                bucket_logger.debug("Waiting for workers to perform evaluation")
                future_all_eval_stats_before = pool.map_async(call, [
                    partial(
                        process_in_batches,
                        batch_size=eval_batch_size,
                        model=model,
                        batch_processor=evaluator,
                        edges=edges,
                        indices=eval_edge_perm[s],
                    )
                    for s in split_almost_equally(eval_edge_perm.size(0),
                                                  num_parts=num_workers)
                ])
                all_eval_stats_before = \
                    get_async_result(future_all_eval_stats_before, pool)
                eval_stats_before = Stats.sum(all_eval_stats_before).average()
                bucket_logger.info(f"Stats before training: {eval_stats_before}")

            io_time += time.time() - tic
            tic = time.time()
            # HOGWILD training
            bucket_logger.debug("Waiting for workers to perform training")
            # FIXME should we only delay if iteration_idx == 0?
            future_all_stats = pool.map_async(call, [
                partial(
                    process_in_batches,
                    batch_size=config.batch_size,
                    model=model,
                    batch_processor=trainer,
                    edges=edges,
                    indices=edge_perm[s],
                    delay=config.hogwild_delay if epoch_idx == 0 and rank > 0 else 0,
                )
                for rank, s in enumerate(split_almost_equally(edge_perm.size(0),
                                                              num_parts=num_workers))
            ])
            all_stats = get_async_result(future_all_stats, pool)
            stats = Stats.sum(all_stats).average()
            compute_time = time.time() - tic

            bucket_logger.info(
                f"bucket {total_buckets - remaining} / {total_buckets} : "
                f"Processed {num_edges} edges in {compute_time:.2f} s "
                f"( {num_edges / compute_time / 1e6:.2g} M/sec ); "
                f"io: {io_time:.2f} s ( {io_bytes / io_time / 1e6:.2f} MB/sec )")
            bucket_logger.info(f"{stats}")

            # HOGWILD eval after training
            eval_stats_after: Optional[Stats] = None
            if num_eval_edges > 0:
                bucket_logger.debug("Waiting for workers to perform evaluation")
                future_all_eval_stats_after = pool.map_async(call, [
                    partial(
                        process_in_batches,
                        batch_size=eval_batch_size,
                        model=model,
                        batch_processor=evaluator,
                        edges=edges,
                        indices=eval_edge_perm[s],
                    )
                    for s in split_almost_equally(eval_edge_perm.size(0),
                                                  num_parts=num_workers)
                ])
                all_eval_stats_after = \
                    get_async_result(future_all_eval_stats_after, pool)
                eval_stats_after = Stats.sum(all_eval_stats_after).average()
                bucket_logger.info(f"Stats after training: {eval_stats_after}")

            # Add train/eval metrics to queue
            yield current_index, eval_stats_before, stats, eval_stats_after

        swap_partitioned_embeddings(cur_b, None)

        # Distributed Processing: all machines can leave the barrier now.
        sync.barrier()

        # Preserving a checkpoint requires two steps:
        # - create a snapshot (w/ symlinks) after it's first written;
        # - don't delete it once the following one is written.
        # These two happen in two successive iterations of the main loop: the
        # one just before and the one just after the epoch boundary.
        preserve_old_checkpoint = should_preserve_old_checkpoint(
            iteration_manager, config.checkpoint_preservation_interval)
        preserve_new_checkpoint = should_preserve_old_checkpoint(
            iteration_manager + 1, config.checkpoint_preservation_interval)

        # Write metadata: for multiple machines, write from rank-0
        logger.info(
            f"Finished epoch {epoch_idx + 1} / {iteration_manager.num_epochs}, "
            f"edge path {edge_path_idx + 1} / {iteration_manager.num_edge_paths}, "
            f"edge chunk {edge_chunk_idx + 1} / {iteration_manager.num_edge_chunks}")
        if rank == 0:
            for entity, econfig in config.entities.items():
                if econfig.num_partitions == 1:
                    embs = model.get_embeddings(entity, Side.LHS)
                    optimizer = trainer.entity_optimizers[(entity, Partition(0))]

                    checkpoint_manager.write(
                        entity, Partition(0),
                        embs.detach(), OptimizerStateDict(optimizer.state_dict()))

            sanitized_state_dict: ModuleStateDict = {}
            for k, v in ModuleStateDict(model.state_dict()).items():
                if k.startswith('lhs_embs') or k.startswith('rhs_embs'):
                    # skipping state that's an entity embedding
                    continue
                sanitized_state_dict[k] = v

            logger.info("Writing the metadata")
            checkpoint_manager.write_model(
                sanitized_state_dict,
                OptimizerStateDict(trainer.global_optimizer.state_dict()),
            )

        logger.info("Writing the checkpoint")
        checkpoint_manager.write_new_version(config)

        dist_logger.info("Waiting for other workers to write their parts of the checkpoint")
        sync.barrier()
        dist_logger.info("All parts of the checkpoint have been written")

        logger.info("Switching to the new checkpoint version")
        checkpoint_manager.switch_to_new_version()

        dist_logger.info("Waiting for other workers to switch to the new checkpoint version")
        sync.barrier()
        dist_logger.info("All workers have switched to the new checkpoint version")

        # After all the machines have finished committing
        # checkpoints, we either remove the old checkpoints
        # or we preserve it
        if preserve_new_checkpoint:
            # Add 1 so the index is a multiple of the interval, it looks nicer.
            checkpoint_manager.preserve_current_version(config, epoch_idx + 1)
        if not preserve_old_checkpoint:
            checkpoint_manager.remove_old_version(config)

        # now we're sure that all partition files exist,
        # so be strict about loading them
        strict = True

    # quiescence
    pool.close()
    pool.join()

    sync.barrier()

    checkpoint_manager.close()
    if loadpath_manager is not None:
        loadpath_manager.close()

    # FIXME join distributed workers (not really necessary)

    logger.info("Exiting")


def train(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    trainer: Optional[AbstractBatchProcessor] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    rank: Rank = RANK_ZERO,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> None:
    # Create and run the generator until exhaustion.
    for _ in train_and_report_stats(config, model, trainer, evaluator, rank, subprocess_init):
        pass


def main():
    setup_logging()
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
    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, overrides)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    train(config, rank=Rank(opt.rank), subprocess_init=subprocess_init)


if __name__ == '__main__':
    main()
