#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from abc import abstractmethod
from itertools import chain
from typing import Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

import attr
import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from .bucket_scheduling import create_buckets_ordered_lexicographically
from .config import parse_config, ConfigSchema
from .entitylist import EntityList
from .fileio import CheckpointManager, EdgeReader
from .model import RankingLoss, make_model, override_model, MultiRelationEmbedder, \
    Margins, Scores
from .stats import Stats, stats
from .types import Side, Rank, Bucket, EntityName, Partition, FloatTensorType, \
    LongTensorType
from .util import log, get_partitioned_types, chunk_by_index, create_pool, \
    compute_randomized_auc, split_almost_equally, get_num_workers


StatsType = TypeVar("StatsType", bound=Stats)


class AbstractEvaluator(Generic[StatsType]):

    @abstractmethod
    def eval(
        self,
        scores: Scores,
        margins: Margins,
        batch_lhs: Union[FloatTensorType, TensorList],
        batch_rhs: Union[FloatTensorType, TensorList],
        batch_rel: Union[int, LongTensorType],
    ) -> StatsType:
        pass

    # This is needed because there's no nice way to retrieve, from an evaluator
    # object, the type of the stats it will return.
    @abstractmethod
    def sum_stats(self, stats: Iterable[StatsType]) -> StatsType:
        """Helper method to do the sum on the right type."""
        pass


@stats
class EvalStats(Stats):
    pos_rank: float = attr.ib()
    mrr: float = attr.ib()
    r1: float = attr.ib()
    r10: float = attr.ib()
    r50: float = attr.ib()
    auc: float = attr.ib()


class RankingEvaluator(AbstractEvaluator[EvalStats]):

    def eval(
        self,
        scores: Scores,
        margins: Margins,
        batch_lhs: Union[FloatTensorType, TensorList],
        batch_rhs: Union[FloatTensorType, TensorList],
        batch_rel: Union[int, LongTensorType],
    ) -> EvalStats:
        batch_size = batch_lhs.size(0)

        # lhs_margin[i,j] is the score of (neg_lhs[i,j], rel[i], rhs[i]) minus the
        # score of (lhs[i], rel[i], rhs[i]). Thus it is >= 0 when the i-th positive
        # edge achieved a lower score than the j-th negative that we sampled for it.
        # remember, the self-interaction has margin=0, but we do want to
        # count any other negatives with margin=0 as violators, so
        # lets make it a >=0 condition and then just assume one extra
        # violator
        lrank = margins[0].ge(0).float().sum(1) + 1
        rrank = margins[1].ge(0).float().sum(1) + 1

        auc = .5 * compute_randomized_auc(scores[0], scores[2], batch_size) \
            + .5 * compute_randomized_auc(scores[1], scores[3], batch_size)

        return EvalStats(
            pos_rank=(lrank.float().sum().item() + rrank.float().sum().item()) / 2,
            mrr=(lrank.float().reciprocal().sum().item()
                 + rrank.float().reciprocal().sum().item()) / 2,
            r1=(lrank.le(1).float() + rrank.le(1).float()).sum().item() / 2,
            r10=(lrank.le(10).float() + rrank.le(10).float()).sum().item() / 2,
            r50=(lrank.le(50).float() + rrank.le(50).float()).sum().item() / 2,
            auc=auc * batch_size,  # at the end the auc will be averaged over count
            count=batch_size)

    def sum_stats(self, stats: Iterable[EvalStats]) -> EvalStats:
        """Helper method to do the sum on the right type."""
        return EvalStats.sum(stats)


DEFAULT_EVALUATOR = RankingEvaluator()


def eval_one_batch(
    model: MultiRelationEmbedder,
    batch_lhs: EntityList,
    batch_rhs: EntityList,
    # batch_rel is int in normal mode, LongTensor in dynamic relations mode.
    batch_rel: Union[int, LongTensorType],
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR,
) -> StatsType:
    batch_lhs = batch_lhs.collapse(model.is_featurized(batch_rel, Side.LHS))
    batch_rhs = batch_rhs.collapse(model.is_featurized(batch_rel, Side.RHS))
    # For evaluation, we want the ranking, not margin loss, so set
    # loss_fn = "ranking"
    # margin = 0
    with override_model(model, loss_fn=RankingLoss(0)):
        loss, margins, scores = model(batch_lhs, batch_rhs, batch_rel)

    return evaluator.eval(scores, margins, batch_lhs, batch_rhs, batch_rel)


def eval_many_batches(
    config: ConfigSchema,
    model: MultiRelationEmbedder,
    lhs: EntityList,
    rhs: EntityList,
    rel: LongTensorType,
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR,
) -> StatsType:
    all_stats = []

    # FIXME: it's not really safe to do partial batches if num_batch_negs != 0
    # because partial batches will produce incorrect results, and if the
    # dataset per thread is very small then every batch may be partial. I don't
    # know if a perfect solution for this that doesn't introduce other biases...

    if model.num_dynamic_rels > 0:
        offset, num_edges = 0, rel.size(0)
        while offset < num_edges:
            batch_size = min(num_edges - offset, config.batch_size)
            all_stats.append(eval_one_batch(
                model,
                lhs[offset:offset + batch_size],
                rhs[offset:offset + batch_size],
                rel[offset:offset + batch_size],
                evaluator))
            offset += batch_size
    else:
        _, lhs_chunked, rhs_chunked = chunk_by_index(rel, lhs, rhs)
        batch_size = config.batch_size
        for rel_type, (lhs_rel, rhs_rel) in enumerate(zip(lhs_chunked, rhs_chunked)):
            if lhs_rel.nelement() == 0:
                continue

            for offset in range(0, lhs_rel.size(0), batch_size):
                batch_lhs = lhs_rel[offset:offset + batch_size]
                batch_rhs = rhs_rel[offset:offset + batch_size]
                all_stats.append(
                    eval_one_batch(model, batch_lhs, batch_rhs, rel_type, evaluator))

    return evaluator.sum_stats(all_stats)


def eval_one_thread(
    rank: Rank,
    config: ConfigSchema,
    model: MultiRelationEmbedder,
    lhs: EntityList,
    rhs: EntityList,
    rel: LongTensorType,
    evaluator: AbstractEvaluator[StatsType],
) -> StatsType:
    """ This is the eval loop executed by each HOGWILD thread.
    """
    stats = eval_many_batches(config, model, lhs, rhs, rel, evaluator)
    # print("Rank %d done" % rank)
    return stats


def do_eval_and_report_stats(
    config: ConfigSchema,
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR,
) -> Generator[Tuple[Optional[int], Optional[Bucket], StatsType], None, None]:
    """Computes eval metrics (mr/mrr/r1/r10/r50) for a checkpoint with trained
       embeddings.
    """

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
    pool = create_pool(num_workers)

    model = make_model(config)

    state_dict, _ = checkpoint_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    for entity, econfig in config.entities.items():
        if econfig.num_partitions == 1:
            embs = load_embeddings(entity, Partition(0))
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)

    all_stats: List[EvalStats] = []
    for edge_path_idx, edge_path in enumerate(config.edge_paths):
        log("Starting edge path %d / %d (%s)"
            % (edge_path_idx + 1, len(config.edge_paths), edge_path))
        edge_reader = EdgeReader(edge_path)

        all_edge_path_stats = []
        last_lhs, last_rhs = None, None
        for bucket in create_buckets_ordered_lexicographically(nparts_lhs, nparts_rhs):
            tic = time.time()
            # log("%s: Loading entities" % (bucket,))

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

            # log("%s: Loading edges" % (bucket,))
            lhs, rhs, rel = edge_reader.read(bucket.lhs, bucket.rhs)
            num_edges = rel.size(0)

            load_time = time.time() - tic
            tic = time.time()
            # log("%s: Launching and waiting for workers" % (bucket,))
            all_bucket_stats = pool.starmap(eval_one_thread, [
                (Rank(i), config, model, lhs[s], rhs[s], rel[s], evaluator)
                for i, s in enumerate(split_almost_equally(num_edges, num_parts=num_workers))
            ])

            compute_time = time.time() - tic
            log("%s: Processed %d edges in %.2g s (%.2gM/sec); load time: %.2g s"
                % (bucket, lhs.size(0), compute_time,
                   lhs.size(0) / compute_time / 1e6, load_time))

            total_bucket_stats = evaluator.sum_stats(all_bucket_stats)
            all_edge_path_stats.append(total_bucket_stats)
            mean_bucket_stats = total_bucket_stats.average()
            log("Stats for edge path %d / %d, bucket %s: %s"
                % (edge_path_idx + 1, len(config.edge_paths), bucket,
                   mean_bucket_stats))

            yield edge_path_idx, bucket, mean_bucket_stats

        total_edge_path_stats = evaluator.sum_stats(all_edge_path_stats)
        all_stats.append(total_edge_path_stats)
        mean_edge_path_stats = total_edge_path_stats.average()
        log("")
        log("Stats for edge path %d / %d: %s"
            % (edge_path_idx + 1, len(config.edge_paths), mean_edge_path_stats))
        log("")

        yield edge_path_idx, None, mean_edge_path_stats

    mean_stats = evaluator.sum_stats(all_stats).average()
    log("")
    log("Stats: %s" % mean_stats)
    log("")

    yield None, None, mean_stats

    pool.close()
    pool.join()


def do_eval(
    config: ConfigSchema,
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR
) -> None:
    # Create and run the generator until exhaustion.
    for _ in do_eval_and_report_stats(config, evaluator):
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
    opt = parser.parse_args()

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
    else:
        overrides = None
    config = parse_config(opt.config, overrides)

    do_eval(config)


if __name__ == '__main__':
    main()
