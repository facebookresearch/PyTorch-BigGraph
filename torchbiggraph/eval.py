#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
import time
from functools import partial
from itertools import chain
from typing import Callable, Generator, List, Optional, Tuple

import torch

from torchbiggraph.batching import (
    AbstractBatchProcessor,
    call,
    process_in_batches,
)
from torchbiggraph.bucket_scheduling import (
    create_buckets_ordered_lexicographically
)
from torchbiggraph.checkpoint_manager import CheckpointManager
from torchbiggraph.config import add_to_sys_path, ConfigFileLoader, ConfigSchema
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.edgelist_reader import EDGELIST_READERS
from torchbiggraph.model import MultiRelationEmbedder, Scores, make_model
from torchbiggraph.stats import Stats, average_of_sums
from torchbiggraph.types import Bucket, EntityName, Partition, Side
from torchbiggraph.util import (
    compute_randomized_auc,
    create_pool,
    get_async_result,
    get_num_workers,
    get_partitioned_types,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    SubprocessInitializer,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")


class RankingEvaluator(AbstractBatchProcessor):

    def process_one_batch(
        self,
        model: MultiRelationEmbedder,
        batch_edges: EdgeList,
    ) -> Stats:
        with torch.no_grad():
            scores = model(batch_edges)
        return self.eval(scores, batch_edges)

    def eval(
        self,
        scores: Scores,
        batch_edges: EdgeList,
    ) -> Stats:
        batch_size = len(batch_edges)

        lhs_rank = (scores.lhs_neg >= scores.lhs_pos.unsqueeze(1)).sum(1) + 1
        rhs_rank = (scores.rhs_neg >= scores.rhs_pos.unsqueeze(1)).sum(1) + 1

        lhs_auc = compute_randomized_auc(scores.lhs_pos, scores.lhs_neg, batch_size)
        rhs_auc = compute_randomized_auc(scores.rhs_pos, scores.rhs_neg, batch_size)

        return Stats(
            pos_rank=average_of_sums(lhs_rank, rhs_rank),
            mrr=average_of_sums(lhs_rank.float().reciprocal(),
                                rhs_rank.float().reciprocal()),
            r1=average_of_sums(lhs_rank.le(1), rhs_rank.le(1)),
            r10=average_of_sums(lhs_rank.le(10), rhs_rank.le(10)),
            r50=average_of_sums(lhs_rank.le(50), rhs_rank.le(50)),
            # At the end the AUC will be averaged over count.
            auc=batch_size * (lhs_auc + rhs_auc) / 2,
            count=batch_size)


def do_eval_and_report_stats(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> Generator[Tuple[Optional[int], Optional[Bucket], Stats], None, None]:
    """Computes eval metrics (mr/mrr/r1/r10/r50) for a checkpoint with trained
       embeddings.
    """
    tag_logs_with_process_name(f"Evaluator")

    if evaluator is None:
        evaluator = RankingEvaluator()

    if config.verbose > 0:
        import pprint
        pprint.PrettyPrinter().pprint(config.to_dict())

    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    def load_embeddings(entity: EntityName, part: Partition) -> torch.nn.Parameter:
        embs, _ = checkpoint_manager.read(entity, part)
        assert embs.is_shared()
        return torch.nn.Parameter(embs)

    nparts_lhs, lhs_partitioned_types = get_partitioned_types(config, Side.LHS)
    nparts_rhs, rhs_partitioned_types = get_partitioned_types(config, Side.RHS)

    num_workers = get_num_workers(config.workers)
    pool = create_pool(
        num_workers,
        subprocess_name="EvalWorker",
        subprocess_init=subprocess_init,
    )

    if model is None:
        model = make_model(config)
    model.share_memory()

    state_dict, _ = checkpoint_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    for entity, econfig in config.entities.items():
        if econfig.num_partitions == 1:
            embs = load_embeddings(entity, Partition(0))
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)

    all_stats: List[Stats] = []
    for edge_path_idx, edge_path in enumerate(config.edge_paths):
        logger.info(
            f"Starting edge path {edge_path_idx + 1} / {len(config.edge_paths)} "
            f"({edge_path})")
        edgelist_reader = EDGELIST_READERS.make_instance(edge_path)

        all_edge_path_stats = []
        last_lhs, last_rhs = None, None
        for bucket in create_buckets_ordered_lexicographically(nparts_lhs, nparts_rhs):
            tic = time.time()
            # logger.info(f"{bucket}: Loading entities")

            if last_lhs != bucket.lhs:
                for e in lhs_partitioned_types:
                    model.clear_embeddings(e, Side.LHS)
                    embs = load_embeddings(e, bucket.lhs)
                    model.set_embeddings(e, embs, Side.LHS)
            if last_rhs != bucket.rhs:
                for e in rhs_partitioned_types:
                    model.clear_embeddings(e, Side.RHS)
                    embs = load_embeddings(e, bucket.rhs)
                    model.set_embeddings(e, embs, Side.RHS)
            last_lhs, last_rhs = bucket.lhs, bucket.rhs

            # logger.info(f"{bucket}: Loading edges")
            edges = edgelist_reader.read_edgelist(bucket.lhs, bucket.rhs)
            num_edges = len(edges)

            load_time = time.time() - tic
            tic = time.time()
            # logger.info(f"{bucket}: Launching and waiting for workers")
            future_all_bucket_stats = pool.map_async(call, [
                partial(
                    process_in_batches,
                    batch_size=config.batch_size,
                    model=model,
                    batch_processor=evaluator,
                    edges=edges[s],
                )
                for s in split_almost_equally(num_edges, num_parts=num_workers)
            ])
            all_bucket_stats = \
                get_async_result(future_all_bucket_stats, pool)

            compute_time = time.time() - tic
            logger.info(
                f"{bucket}: Processed {num_edges} edges in {compute_time:.2g} s "
                f"({num_edges / compute_time / 1e6:.2g}M/sec); "
                f"load time: {load_time:.2g} s")

            total_bucket_stats = Stats.sum(all_bucket_stats)
            all_edge_path_stats.append(total_bucket_stats)
            mean_bucket_stats = total_bucket_stats.average()
            logger.info(
                f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}, "
                f"bucket {bucket}: {mean_bucket_stats}")

            yield edge_path_idx, bucket, mean_bucket_stats

        total_edge_path_stats = Stats.sum(all_edge_path_stats)
        all_stats.append(total_edge_path_stats)
        mean_edge_path_stats = total_edge_path_stats.average()
        logger.info("")
        logger.info(
            f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}: "
            f"{mean_edge_path_stats}")
        logger.info("")

        yield edge_path_idx, None, mean_edge_path_stats

    mean_stats = Stats.sum(all_stats).average()
    logger.info("")
    logger.info(f"Stats: {mean_stats}")
    logger.info("")

    yield None, None, mean_stats

    pool.close()
    pool.join()


def do_eval(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    evaluator: Optional[AbstractBatchProcessor] = None,
    subprocess_init: Optional[Callable[[], None]] = None,
) -> None:
    # Create and run the generator until exhaustion.
    for _ in do_eval_and_report_stats(config, model, evaluator, subprocess_init):
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

    do_eval(config, subprocess_init=subprocess_init)


if __name__ == '__main__':
    main()
