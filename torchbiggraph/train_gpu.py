#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import ctypes
import logging
import os
import time
from collections import defaultdict
from multiprocessing.connection import wait as mp_wait
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
from torchbiggraph.batching import AbstractBatchProcessor, process_in_batches
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.train_cpu import Trainer, TrainingCoordinator
from torchbiggraph.types import (
    SINGLE_TRAINER,
    Bucket,
    EntityName,
    FloatTensorType,
    GPURank,
    LongTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
    Side,
    SubPartition,
)
from torchbiggraph.util import (
    BucketLogger,
    DummyOptimizer,
    EmbeddingHolder,
    SubprocessInitializer,
    allocate_shared_tensor,
    create_pool,
    div_roundup,
    fast_approx_rand,
    get_async_result,
    get_num_workers,
    hide_distributed_logging,
    round_up_to_nearest_multiple,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    tag_logs_with_process_name,
)


try:
    from torchbiggraph import _C

    CPP_INSTALLED = True
except ImportError:
    CPP_INSTALLED = False


logger = logging.getLogger("torchbiggraph")
dist_logger = logging.LoggerAdapter(logger, {"distributed": True})


class TimeKeeper:
    def __init__(self):
        self.t = self._get_time()
        self.sub_ts = {}

    def _get_time(self) -> float:
        return time.monotonic_ns() / 1e9

    def start(self, name: str) -> None:
        self.sub_ts[name] = self._get_time()

    def stop(self, name: str) -> float:
        start_t = self.sub_ts.pop(name)
        stop_t = self._get_time()
        delta_t = stop_t - start_t
        self.t += delta_t
        return delta_t

    def unaccounted(self) -> float:
        t = self._get_time()
        return t - self.t


class SubprocessArgs(NamedTuple):
    lhs_types: Set[str]
    rhs_types: Set[str]
    lhs_part: Partition
    rhs_part: Partition
    lhs_subpart: SubPartition
    rhs_subpart: SubPartition
    next_lhs_subpart: Optional[SubPartition]
    next_rhs_subpart: Optional[SubPartition]
    model: MultiRelationEmbedder
    trainer: Trainer
    all_embs: Dict[Tuple[EntityName, Partition], FloatTensorType]
    subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice]
    subbuckets: Dict[
        Tuple[int, int], Tuple[LongTensorType, LongTensorType, LongTensorType]
    ]
    batch_size: int
    lr: float


class SubprocessReturn(NamedTuple):
    gpu_idx: GPURank
    stats: Stats


class GPUProcess(mp.get_context("spawn").Process):
    def __init__(
        self,
        gpu_idx: GPURank,
        subprocess_init: Optional[Callable[[], None]] = None,
        embedding_storage_freelist: Optional[Set[torch.FloatStorage]] = None,
    ) -> None:
        super().__init__(daemon=True, name=f"GPU #{gpu_idx}")
        self.gpu_idx = gpu_idx
        self.master_endpoint, self.worker_endpoint = mp.get_context("spawn").Pipe()
        self.subprocess_init = subprocess_init
        self.sub_holder: Dict[
            Tuple[EntityName, Partition, SubPartition],
            Tuple[torch.nn.Parameter, RowAdagrad],
        ] = {}
        self.embedding_storage_freelist = embedding_storage_freelist

    @property
    def my_device(self) -> torch.device:
        return torch.device("cuda", index=self.gpu_idx)

    def run(self) -> None:
        torch.set_num_threads(1)
        torch.cuda.set_device(self.my_device)
        if self.subprocess_init is not None:
            self.subprocess_init()
        self.master_endpoint.close()
        for s in self.embedding_storage_freelist:
            assert s.is_shared()
            cptr = ctypes.c_void_p(s.data_ptr())
            csize = ctypes.c_size_t(s.size() * s.element_size())
            cflags = ctypes.c_uint(0)
            # FIXME: broken by D20249187
            # cudart = torch.cuda.cudart()
            cudart = ctypes.cdll.LoadLibrary(None)
            res = cudart.cudaHostRegister(cptr, csize, cflags)
            torch.cuda.check_error(res)
            assert s.is_pinned()
        logger.info(f"GPU subprocess {self.gpu_idx} up and running")
        while True:
            try:
                job: SubprocessArgs = self.worker_endpoint.recv()
            except EOFError:
                break

            stats = self.do_one_job(
                lhs_types=job.lhs_types,
                rhs_types=job.rhs_types,
                lhs_part=job.lhs_part,
                rhs_part=job.rhs_part,
                lhs_subpart=job.lhs_subpart,
                rhs_subpart=job.rhs_subpart,
                next_lhs_subpart=job.next_lhs_subpart,
                next_rhs_subpart=job.next_rhs_subpart,
                model=job.model,
                trainer=job.trainer,
                all_embs=job.all_embs,
                subpart_slices=job.subpart_slices,
                subbuckets=job.subbuckets,
                batch_size=job.batch_size,
                lr=job.lr,
            )

            self.worker_endpoint.send(
                SubprocessReturn(gpu_idx=self.gpu_idx, stats=stats)
            )

    def do_one_job(  # noqa
        self,
        lhs_types: Set[str],
        rhs_types: Set[str],
        lhs_part: Partition,
        rhs_part: Partition,
        lhs_subpart: SubPartition,
        rhs_subpart: SubPartition,
        next_lhs_subpart: Optional[SubPartition],
        next_rhs_subpart: Optional[SubPartition],
        model: MultiRelationEmbedder,
        trainer: Trainer,
        all_embs: Dict[Tuple[EntityName, Partition], FloatTensorType],
        subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice],
        subbuckets: Dict[
            Tuple[int, int], Tuple[LongTensorType, LongTensorType, LongTensorType]
        ],
        batch_size: int,
        lr: float,
    ) -> Stats:
        tk = TimeKeeper()

        for embeddings in all_embs.values():
            assert embeddings.is_pinned()

        occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        for entity_name in lhs_types:
            occurrences[entity_name, lhs_part, lhs_subpart].add(Side.LHS)
        for entity_name in rhs_types:
            occurrences[entity_name, rhs_part, rhs_subpart].add(Side.RHS)

        if lhs_part != rhs_part:  # Bipartite
            assert all(len(v) == 1 for v in occurrences.values())

        tk.start("copy_to_device")
        for entity_name, part, subpart in occurrences.keys():
            if (entity_name, part, subpart) in self.sub_holder:
                continue
            embeddings = all_embs[entity_name, part]
            optimizer = trainer.partitioned_optimizers[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            # TODO have two permanent storages on GPU and move stuff in and out
            # from them
            # logger.info(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * embeddings.shape[1] * 4:,} bytes")
            gpu_embeddings = torch.empty(
                (subpart_slice.stop - subpart_slice.start, embeddings.shape[1]),
                dtype=torch.float32,
                device=self.my_device,
            )
            gpu_embeddings.copy_(embeddings[subpart_slice], non_blocking=True)
            gpu_embeddings = torch.nn.Parameter(gpu_embeddings)
            gpu_optimizer = RowAdagrad([gpu_embeddings], lr=lr)
            (cpu_state,) = optimizer.state.values()
            (gpu_state,) = gpu_optimizer.state.values()
            # logger.info(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * 4:,} bytes")
            gpu_state["sum"].copy_(cpu_state["sum"][subpart_slice], non_blocking=True)

            self.sub_holder[entity_name, part, subpart] = (
                gpu_embeddings,
                gpu_optimizer,
            )
        logger.debug(
            f"Time spent copying subparts to GPU: {tk.stop('copy_to_device'):.4f} s"
        )

        for (
            (entity_name, part, subpart),
            (gpu_embeddings, gpu_optimizer),
        ) in self.sub_holder.items():
            for side in occurrences[entity_name, part, subpart]:
                model.set_embeddings(entity_name, side, gpu_embeddings)
                trainer.partitioned_optimizers[
                    entity_name, part, subpart
                ] = gpu_optimizer

        tk.start("translate_edges")
        num_edges = subbuckets[lhs_subpart, rhs_subpart][0].shape[0]
        edge_perm = torch.randperm(num_edges)
        edges_lhs, edges_rhs, edges_rel = subbuckets[lhs_subpart, rhs_subpart]
        _C.shuffle(edges_lhs, edge_perm, os.cpu_count())
        _C.shuffle(edges_rhs, edge_perm, os.cpu_count())
        _C.shuffle(edges_rel, edge_perm, os.cpu_count())
        assert edges_lhs.is_pinned()
        assert edges_rhs.is_pinned()
        assert edges_rel.is_pinned()
        gpu_edges = EdgeList(
            EntityList.from_tensor(edges_lhs),
            EntityList.from_tensor(edges_rhs),
            edges_rel,
        ).to(self.my_device, non_blocking=True)
        logger.debug(f"GPU #{self.gpu_idx} got {num_edges} edges")
        logger.debug(
            f"Time spent copying edges to GPU: {tk.stop('translate_edges'):.4f} s"
        )

        tk.start("processing")
        stats = process_in_batches(
            batch_size=batch_size, model=model, batch_processor=trainer, edges=gpu_edges
        )
        logger.debug(f"Time spent processing: {tk.stop('processing'):.4f} s")

        next_occurrences: Dict[
            Tuple[EntityName, Partition, SubPartition], Set[Side]
        ] = defaultdict(set)
        if next_lhs_subpart is not None:
            for entity_name in lhs_types:
                next_occurrences[entity_name, lhs_part, next_lhs_subpart].add(Side.LHS)
        if next_rhs_subpart is not None:
            for entity_name in rhs_types:
                next_occurrences[entity_name, rhs_part, next_rhs_subpart].add(Side.RHS)

        tk.start("copy_from_device")
        for (entity_name, part, subpart), (gpu_embeddings, gpu_optimizer) in list(
            self.sub_holder.items()
        ):
            if (entity_name, part, subpart) in next_occurrences:
                continue
            embeddings = all_embs[entity_name, part]
            optimizer = trainer.partitioned_optimizers[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            embeddings[subpart_slice].copy_(gpu_embeddings.detach(), non_blocking=True)
            del gpu_embeddings
            (cpu_state,) = optimizer.state.values()
            (gpu_state,) = gpu_optimizer.state.values()
            cpu_state["sum"][subpart_slice].copy_(gpu_state["sum"], non_blocking=True)
            del gpu_state["sum"]
            del self.sub_holder[entity_name, part, subpart]
        logger.debug(
            f"Time spent copying subparts from GPU: {tk.stop('copy_from_device'):.4f} s"
        )

        logger.debug(f"do_one_job: Time unaccounted for: {tk.unaccounted():.4f} s")

        return stats


class GPUProcessPool:
    def __init__(
        self,
        num_gpus: int,
        subprocess_init: Optional[Callable[[], None]] = None,
        embedding_storage_freelist: Optional[Set[torch.FloatStorage]] = None,
    ) -> None:
        self.processes: List[GPUProcess] = [
            GPUProcess(gpu_idx, subprocess_init, embedding_storage_freelist)
            for gpu_idx in range(num_gpus)
        ]
        for p in self.processes:
            p.start()
            p.worker_endpoint.close()

    @property
    def num_gpus(self):
        return len(self.processes)

    def schedule(self, gpu_idx: GPURank, args: SubprocessArgs) -> None:
        self.processes[gpu_idx].master_endpoint.send(args)

    def wait_for_next(self) -> Tuple[GPURank, SubprocessReturn]:
        all_objects = [p.sentinel for p in self.processes] + [
            p.master_endpoint for p in self.processes
        ]
        ready_objects = mp_wait(all_objects)
        for obj in ready_objects:
            for p in self.processes:
                if obj is p.sentinel:
                    raise RuntimeError(
                        f"GPU worker #{p.gpu_idx} (PID: {p.pid}) terminated "
                        f"unexpectedly with exit code {p.exitcode}"
                    )
                if obj is p.master_endpoint:
                    res = p.master_endpoint.recv()
                    return p.gpu_idx, res

    def close(self):
        pass

    def join(self):
        for p in self.processes:
            p.master_endpoint.close()
            p.join()


def build_nonbipartite_schedule_inner(size: int) -> List[List[int]]:
    if size <= 0 or size % 2 != 0:
        raise ValueError("Bad")
    if size == 2:
        return [[(0, 1), (1, 0)]]
    half = size // 2
    pre = [[(i, (i + j) % half + half) for j in range(half)] for i in range(half)]
    post = [[(i + half, (i + j) % half) for j in range(half)] for i in range(half)]
    mid = build_nonbipartite_schedule_inner(half)
    res = []
    res.extend([pre[i] + mid[i] + post[i] for i in range(half // 2)])
    res.extend(
        [
            pre[i + half // 2]
            + [(x + half, y + half) for x, y in mid[i]]
            + post[i + half // 2]
            for i in range(half // 2)
        ]
    )
    return res


def build_nonbipartite_schedule(size: int) -> List[List[int]]:
    if size == 1:
        return [[(0, 0)]]
    if size <= 0 or size % 2 != 0:
        raise ValueError("Bad")
    half = size // 2
    res = build_nonbipartite_schedule_inner(size)
    return [[(i, i)] + res[i] + [(half + i, half + i)] for i in range(half)]


def build_bipartite_schedule(size: int) -> List[List[int]]:
    return [[(i, (i + j) % size) for j in range(size)] for i in range(size)]


NOOP_STATS_HANDLER = StatsHandler()


class GPUTrainingCoordinator(TrainingCoordinator):
    def __init__(
        self,
        config: ConfigSchema,
        model: Optional[MultiRelationEmbedder] = None,
        trainer: Optional[AbstractBatchProcessor] = None,
        evaluator: Optional[AbstractBatchProcessor] = None,
        rank: Rank = SINGLE_TRAINER,
        subprocess_init: Optional[Callable[[], None]] = None,
        stats_handler: StatsHandler = NOOP_STATS_HANDLER,
    ):

        super().__init__(
            config, model, trainer, evaluator, rank, subprocess_init, stats_handler
        )

        assert config.num_gpus > 0
        if not CPP_INSTALLED:
            raise RuntimeError(
                "GPU support requires C++ installation: "
                "install with C++ support by running "
                "`PBG_INSTALL_CPP=1 pip install .`"
            )

        if config.half_precision:
            for entity in config.entities:
                # need this for tensor cores to work
                assert config.entity_dimension(entity) % 8 == 0
            assert config.batch_size % 8 == 0
            assert config.num_batch_negs % 8 == 0
            assert config.num_uniform_negs % 8 == 0

        assert len(self.holder.lhs_unpartitioned_types) == 0
        assert len(self.holder.rhs_unpartitioned_types) == 0

        num_edge_chunks = self.iteration_manager.num_edge_chunks
        max_edges = 0
        for edge_path in config.edge_paths:
            edge_storage = EDGE_STORAGES.make_instance(edge_path)
            for lhs_part in range(self.holder.nparts_lhs):
                for rhs_part in range(self.holder.nparts_rhs):
                    num_edges = edge_storage.get_number_of_edges(lhs_part, rhs_part)
                    num_edges_per_chunk = div_roundup(num_edges, num_edge_chunks)
                    max_edges = max(max_edges, num_edges_per_chunk)
        self.shared_lhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
        self.shared_rhs = allocate_shared_tensor((max_edges,), dtype=torch.long)
        self.shared_rel = allocate_shared_tensor((max_edges,), dtype=torch.long)

        # fork early for HOGWILD threads
        logger.info("Creating GPU workers...")
        torch.set_num_threads(1)
        self.gpu_pool = GPUProcessPool(
            config.num_gpus,
            subprocess_init,
            {s for ss in self.embedding_storage_freelist.values() for s in ss}
            | {
                self.shared_lhs.storage(),
                self.shared_rhs.storage(),
                self.shared_rel.storage(),
            },
        )

    # override
    def _coordinate_train(self, edges, eval_edge_idxs, epoch_idx) -> Stats:
        tk = TimeKeeper()

        config = self.config
        holder = self.holder
        cur_b = self.cur_b
        bucket_logger = self.bucket_logger
        num_edges = len(edges)
        if cur_b.lhs == cur_b.rhs and config.num_gpus > 1:
            num_subparts = 2 * config.num_gpus
        else:
            num_subparts = config.num_gpus

        edges_lhs = edges.lhs.tensor
        edges_rhs = edges.rhs.tensor
        edges_rel = edges.rel
        if eval_edge_idxs is not None:
            bucket_logger.debug("Removing eval edges")
            tk.start("remove_eval")
            num_eval_edges = len(eval_edge_idxs)
            edges_lhs[eval_edge_idxs] = edges_lhs[-num_eval_edges:].clone()
            edges_rhs[eval_edge_idxs] = edges_rhs[-num_eval_edges:].clone()
            edges_rel[eval_edge_idxs] = edges_rel[-num_eval_edges:].clone()
            edges_lhs = edges_lhs[:-num_eval_edges]
            edges_rhs = edges_rhs[:-num_eval_edges]
            edges_rel = edges_rel[:-num_eval_edges]
            bucket_logger.debug(
                f"Time spent removing eval edges: {tk.stop('remove_eval'):.4f} s"
            )

        bucket_logger.debug("Splitting edges into sub-buckets")
        tk.start("mapping_edges")
        # randomly permute the entities, to get a random subbucketing
        perm_holder = {}
        rev_perm_holder = {}
        for (entity, part), embs in holder.partitioned_embeddings.items():
            perm = _C.randperm(self.entity_counts[entity][part], os.cpu_count())
            _C.shuffle(embs, perm, os.cpu_count())
            optimizer = self.trainer.partitioned_optimizers[entity, part]
            (optimizer_state,) = optimizer.state.values()
            _C.shuffle(optimizer_state["sum"], perm, os.cpu_count())
            perm_holder[entity, part] = perm
            rev_perm = _C.reverse_permutation(perm, os.cpu_count())
            rev_perm_holder[entity, part] = rev_perm

        subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice] = {}
        for entity_name, part in holder.partitioned_embeddings.keys():
            num_entities = self.entity_counts[entity_name][part]
            for subpart, subpart_slice in enumerate(
                split_almost_equally(num_entities, num_parts=num_subparts)
            ):
                subpart_slices[entity_name, part, subpart] = subpart_slice

        subbuckets = _C.sub_bucket(
            edges_lhs,
            edges_rhs,
            edges_rel,
            [self.entity_counts[r.lhs][cur_b.lhs] for r in config.relations],
            [perm_holder[r.lhs, cur_b.lhs] for r in config.relations],
            [self.entity_counts[r.rhs][cur_b.rhs] for r in config.relations],
            [perm_holder[r.rhs, cur_b.rhs] for r in config.relations],
            self.shared_lhs,
            self.shared_rhs,
            self.shared_rel,
            num_subparts,
            num_subparts,
            os.cpu_count(),
            config.dynamic_relations,
        )
        bucket_logger.debug(
            "Time spent splitting edges into sub-buckets: "
            f"{tk.stop('mapping_edges'):.4f} s"
        )
        bucket_logger.debug("Done splitting edges into sub-buckets")
        bucket_logger.debug(f"{subpart_slices}")

        tk.start("scheduling")
        busy_gpus: Set[int] = set()
        all_stats: List[Stats] = []
        if cur_b.lhs != cur_b.rhs:  # Graph is bipartite!!
            gpu_schedules = build_bipartite_schedule(num_subparts)
        else:
            gpu_schedules = build_nonbipartite_schedule(num_subparts)
        for s in gpu_schedules:
            s.append(None)
            s.append(None)
        index_in_schedule = [0 for _ in range(self.gpu_pool.num_gpus)]
        locked_parts = set()

        def schedule(gpu_idx: GPURank) -> None:
            if gpu_idx in busy_gpus:
                return
            this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
            next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
            if this_bucket is None:
                return
            subparts = {
                (e, cur_b.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
            } | {(e, cur_b.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
            if any(k in locked_parts for k in subparts):
                return
            for k in subparts:
                locked_parts.add(k)
            busy_gpus.add(gpu_idx)
            bucket_logger.debug(
                f"GPU #{gpu_idx} gets {this_bucket[0]}, {this_bucket[1]}"
            )
            for embs in holder.partitioned_embeddings.values():
                assert embs.is_shared()
            self.gpu_pool.schedule(
                gpu_idx,
                SubprocessArgs(
                    lhs_types=holder.lhs_partitioned_types,
                    rhs_types=holder.rhs_partitioned_types,
                    lhs_part=cur_b.lhs,
                    rhs_part=cur_b.rhs,
                    lhs_subpart=this_bucket[0],
                    rhs_subpart=this_bucket[1],
                    next_lhs_subpart=next_bucket[0]
                    if next_bucket is not None
                    else None,
                    next_rhs_subpart=next_bucket[1]
                    if next_bucket is not None
                    else None,
                    trainer=self.trainer,
                    model=self.model,
                    all_embs=holder.partitioned_embeddings,
                    subpart_slices=subpart_slices,
                    subbuckets=subbuckets,
                    batch_size=config.batch_size,
                    lr=config.lr,
                ),
            )

        for gpu_idx in range(self.gpu_pool.num_gpus):
            schedule(gpu_idx)
        while busy_gpus:
            gpu_idx, result = self.gpu_pool.wait_for_next()
            assert gpu_idx == result.gpu_idx
            all_stats.append(result.stats)
            busy_gpus.remove(gpu_idx)
            this_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx]]
            next_bucket = gpu_schedules[gpu_idx][index_in_schedule[gpu_idx] + 1]
            subparts = {
                (e, cur_b.lhs, this_bucket[0]) for e in holder.lhs_partitioned_types
            } | {(e, cur_b.rhs, this_bucket[1]) for e in holder.rhs_partitioned_types}
            for k in subparts:
                locked_parts.remove(k)
            index_in_schedule[gpu_idx] += 1
            if next_bucket is None:
                bucket_logger.debug(f"GPU #{gpu_idx} finished its schedule")
            for gpu_idx in range(config.num_gpus):
                schedule(gpu_idx)

        assert len(all_stats) == num_subparts * num_subparts
        time_spent_scheduling = tk.stop("scheduling")
        bucket_logger.debug(
            f"Time spent scheduling sub-buckets: {time_spent_scheduling:.4f} s"
        )
        bucket_logger.info(f"Speed: {num_edges / time_spent_scheduling:,.0f} edges/sec")

        tk.start("rev_perm")

        for (entity, part), embs in holder.partitioned_embeddings.items():
            rev_perm = rev_perm_holder[entity, part]
            optimizer = self.trainer.partitioned_optimizers[entity, part]
            _C.shuffle(embs, rev_perm, os.cpu_count())
            (state,) = optimizer.state.values()
            _C.shuffle(state["sum"], rev_perm, os.cpu_count())

        bucket_logger.debug(
            f"Time spent mapping embeddings back from sub-buckets: {tk.stop('rev_perm'):.4f} s"
        )

        logger.debug(
            f"_coordinate_train: Time unaccounted for: {tk.unaccounted():.4f} s"
        )

        return Stats.sum(all_stats).average()
