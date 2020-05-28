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
from typing import Callable, Generator, List, Optional, Tuple

import torch
from torchbiggraph.batching import AbstractBatchProcessor, call, process_in_batches
from torchbiggraph.bucket_scheduling import create_buckets_ordered_lexicographically
from torchbiggraph.checkpoint_manager import CheckpointManager
from torchbiggraph.config import ConfigFileLoader, ConfigSchema, add_to_sys_path
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.graph_storages import EDGE_STORAGES
from torchbiggraph.losses import LOSS_FUNCTIONS
from torchbiggraph.model import MultiRelationEmbedder, Scores, make_model
from torchbiggraph.stats import Stats, average_of_sums
from torchbiggraph.types import UNPARTITIONED, Bucket, EntityName, Partition
from torchbiggraph.util import (
    EmbeddingHolder,
    SubprocessInitializer,
    compute_randomized_auc,
    create_pool,
    get_async_result,
    get_num_workers,
    set_logging_verbosity,
    setup_logging,
    split_almost_equally,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")


class RankingEvaluator(AbstractBatchProcessor):
    def _process_one_batch(
        self, model: MultiRelationEmbedder, batch_edges: EdgeList
    ) -> Stats:

        with torch.no_grad():
            scores, _ = model(batch_edges)

        self._adjust_scores(scores, batch_edges)

        batch_size = len(batch_edges)

        loss = self.calc_loss(scores, batch_edges)

        ranks = []
        aucs = []
        if scores.lhs_neg.nelement() > 0:
            lhs_rank = (scores.lhs_neg >= scores.lhs_pos.unsqueeze(1)).sum(1) + 1
            lhs_auc = compute_randomized_auc(scores.lhs_pos, scores.lhs_neg, batch_size)
            ranks.append(lhs_rank)
            aucs.append(lhs_auc)

        if scores.rhs_neg.nelement() > 0:
            rhs_rank = (scores.rhs_neg >= scores.rhs_pos.unsqueeze(1)).sum(1) + 1
            rhs_auc = compute_randomized_auc(scores.rhs_pos, scores.rhs_neg, batch_size)
            ranks.append(rhs_rank)
            aucs.append(rhs_auc)

        return Stats(
            loss=float(loss),
            pos_rank=average_of_sums(*ranks),
            mrr=average_of_sums(*(rank.float().reciprocal() for rank in ranks)),
            r1=average_of_sums(*(rank.le(1) for rank in ranks)),
            r10=average_of_sums(*(rank.le(10) for rank in ranks)),
            r50=average_of_sums(*(rank.le(50) for rank in ranks)),
            # At the end the AUC will be averaged over count.
            auc=batch_size * sum(aucs) / len(aucs),
            count=batch_size,
        )

    def _adjust_scores(self, scores: Scores, batch_edges: EdgeList):
        # This is a hook for the filtered evaluator to do the filtering
        # of true edges
        pass


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
        evaluator = RankingEvaluator(
            loss_fn=LOSS_FUNCTIONS.get_class(config.loss_fn)(margin=config.margin),
            relation_weights=[relation.weight for relation in config.relations],
        )

    if config.verbose > 0:
        import pprint

        pprint.PrettyPrinter().pprint(config.to_dict())

    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    def load_embeddings(entity: EntityName, part: Partition) -> torch.nn.Parameter:
        embs, _ = checkpoint_manager.read(entity, part)
        assert embs.is_shared()
        return torch.nn.Parameter(embs)

    holder = EmbeddingHolder(config)

    num_workers = get_num_workers(config.workers)
    pool = create_pool(
        num_workers, subprocess_name="EvalWorker", subprocess_init=subprocess_init
    )

    if model is None:
        model = make_model(config)
    model.share_memory()

    state_dict, _ = checkpoint_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    for entity in holder.lhs_unpartitioned_types | holder.rhs_unpartitioned_types:
        embs = load_embeddings(entity, UNPARTITIONED)
        holder.unpartitioned_embeddings[entity] = embs

    all_stats: List[Stats] = []
    for edge_path_idx, edge_path in enumerate(config.edge_paths):
        logger.info(
            f"Starting edge path {edge_path_idx + 1} / {len(config.edge_paths)} "
            f"({edge_path})"
        )
        edge_storage = EDGE_STORAGES.make_instance(edge_path)

        all_edge_path_stats = []
        # FIXME This order assumes higher affinity on the left-hand side, as it's
        # the one changing more slowly. Make this adaptive to the actual affinity.
        for bucket in create_buckets_ordered_lexicographically(
            holder.nparts_lhs, holder.nparts_rhs
        ):
            tic = time.perf_counter()
            # logger.info(f"{bucket}: Loading entities")

            old_parts = set(holder.partitioned_embeddings.keys())
            new_parts = {(e, bucket.lhs) for e in holder.lhs_partitioned_types} | {
                (e, bucket.rhs) for e in holder.rhs_partitioned_types
            }
            for entity, part in old_parts - new_parts:
                del holder.partitioned_embeddings[entity, part]
            for entity, part in new_parts - old_parts:
                embs = load_embeddings(entity, part)
                holder.partitioned_embeddings[entity, part] = embs

            model.set_all_embeddings(holder, bucket)

            # logger.info(f"{bucket}: Loading edges")
            edges = edge_storage.load_edges(bucket.lhs, bucket.rhs)
            num_edges = len(edges)

            load_time = time.perf_counter() - tic
            tic = time.perf_counter()
            # logger.info(f"{bucket}: Launching and waiting for workers")
            future_all_bucket_stats = pool.map_async(
                call,
                [
                    partial(
                        process_in_batches,
                        batch_size=config.batch_size,
                        model=model,
                        batch_processor=evaluator,
                        edges=edges[s],
                    )
                    for s in split_almost_equally(num_edges, num_parts=num_workers)
                ],
            )
            all_bucket_stats = get_async_result(future_all_bucket_stats, pool)

            compute_time = time.perf_counter() - tic
            logger.info(
                f"{bucket}: Processed {num_edges} edges in {compute_time:.2g} s "
                f"({num_edges / compute_time / 1e6:.2g}M/sec); "
                f"load time: {load_time:.2g} s"
            )

            total_bucket_stats = Stats.sum(all_bucket_stats)
            all_edge_path_stats.append(total_bucket_stats)
            mean_bucket_stats = total_bucket_stats.average()
            logger.info(
                f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}, "
                f"bucket {bucket}: {mean_bucket_stats}"
            )

            model.clear_all_embeddings()

            yield edge_path_idx, bucket, mean_bucket_stats

        total_edge_path_stats = Stats.sum(all_edge_path_stats)
        all_stats.append(total_edge_path_stats)
        mean_edge_path_stats = total_edge_path_stats.average()
        logger.info("")
        logger.info(
            f"Stats for edge path {edge_path_idx + 1} / {len(config.edge_paths)}: "
            f"{mean_edge_path_stats}"
        )
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
    config_help = "\n\nConfig parameters:\n\n" + "\n".join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    opt = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, opt.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    do_eval(config, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
