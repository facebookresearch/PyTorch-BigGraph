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

from .config import parse_config, ConfigSchema
from .entitylist import EntityList
from .fileio import CheckpointManager, EdgeReader
from .model import RankingLoss, make_model, override_model, MultiRelationEmbedder, \
    Margins, Scores
from .util import log, get_partitioned_types, chunk_by_index, create_pool, \
    compute_randomized_auc, Side, infer_input_index_base, Rank, \
    create_buckets_ordered_lexicographically, Bucket, Partition, \
    split_almost_equally
from .stats import Stats, stats


StatsType = TypeVar("StatsType", bound=Stats)


class AbstractEvaluator(Generic[StatsType]):

    @abstractmethod
    def eval(
        self,
        scores: Scores,
        margins: Margins,
        batch_lhs: Union[torch.FloatTensor, TensorList],
        batch_rhs: Union[torch.FloatTensor, TensorList],
        batch_rel: Union[int, torch.LongTensor],
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
        batch_lhs: Union[torch.FloatTensor, TensorList],
        batch_rhs: Union[torch.FloatTensor, TensorList],
        batch_rel: Union[int, torch.LongTensor],
    ) -> EvalStats:
        B = batch_lhs.size(0)

        # lhs_margin[i,j] is the score of (neg_lhs[i,j], rel[i], rhs[i]) minus the
        # score of (lhs[i], rel[i], rhs[i]). Thus it is >= 0 when the i-th positive
        # edge achieved a lower score than the j-th negative that we sampled for it.
        # remember, the self-interaction has margin=0, but we do want to
        # count any other negatives with margin=0 as violators, so
        # lets make it a >=0 condition and then just assume one extra
        # violator
        lrank = margins[0].ge(0).float().sum(1) + 1
        rrank = margins[1].ge(0).float().sum(1) + 1

        auc = .5 * compute_randomized_auc(scores[0], scores[2], B)\
              + .5 * compute_randomized_auc(scores[1], scores[3], B)

        return EvalStats(
            pos_rank=(lrank.float().sum().item() + rrank.float().sum().item()) / 2,
            mrr=(lrank.float().reciprocal().sum().item()
                 + rrank.float().reciprocal().sum().item()) / 2,
            r1=(lrank.le(1).float() + rrank.le(1).float()).sum().item() / 2,
            r10=(lrank.le(10).float() + rrank.le(10).float()).sum().item() / 2,
            r50=(lrank.le(50).float() + rrank.le(50).float()).sum().item() / 2,
            auc=auc * B,  # at the end the auc will be averaged over count
            count=B)

    def sum_stats(self, stats: Iterable[EvalStats]) -> EvalStats:
        """Helper method to do the sum on the right type."""
        return EvalStats.sum(stats)


DEFAULT_EVALUATOR = RankingEvaluator()


def eval_one_batch(
    model: MultiRelationEmbedder,
    batch_lhs: EntityList,
    batch_rhs: EntityList,
    # batch_rel is int in normal mode, LongTensor in dynamic relations mode.
    batch_rel: Union[int, torch.LongTensor],
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
    rel: torch.LongTensor,
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR,
) -> StatsType:
    all_stats = []
    if model.num_dynamic_rels > 0:
        offset, N = 0, rel.size(0)
        while offset < N:
            B = min(N - offset, config.batch_size)
            all_stats.append(eval_one_batch(
                model,
                lhs[offset:offset + B],
                rhs[offset:offset + B],
                rel[offset:offset + B],
                evaluator))
            offset += B
    else:
        _, lhs_chunked, rhs_chunked = chunk_by_index(rel, lhs, rhs)
        B = config.batch_size
        for rel_type, (lhs_rel, rhs_rel) in enumerate(zip(lhs_chunked, rhs_chunked)):
            if lhs_rel.nelement() == 0:
                continue

            for offset in range(0, lhs_rel.size(0), B):
                batch_lhs = lhs_rel[offset:offset + B]
                batch_rhs = rhs_rel[offset:offset + B]
                all_stats.append(
                    eval_one_batch(model, batch_lhs, batch_rhs, rel_type, evaluator))

    return evaluator.sum_stats(all_stats)


def eval_one_thread(
    rank: Rank,
    config: ConfigSchema,
    model: MultiRelationEmbedder,
    lhs: EntityList,
    rhs: EntityList,
    rel: torch.LongTensor,
    evaluator: AbstractEvaluator[StatsType],
) -> StatsType:
    """ This is the eval loop executed by each HOGWILD thread.
    """
    stats = eval_many_batches(config, model, lhs, rhs, rel, evaluator)
    print("Rank %d done" % rank)
    return stats


def do_eval_and_report_stats(
    config: ConfigSchema,
    evaluator: AbstractEvaluator[StatsType] = DEFAULT_EVALUATOR,
) -> Generator[Tuple[Optional[int], Optional[Bucket], StatsType], None, None]:
    """Computes eval metrics (r1/r10/r50) for a checkpoint with trained
       embeddings.
    """

    index_base = infer_input_index_base(config)

    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    def load_embeddings(entity, part: Partition = 0):
        data = checkpoint_manager.read(entity, part, strict=True)
        embs, _optim_state = data
        embs.share_memory_()
        if not isinstance(embs, torch.nn.Parameter):  # Pytorch bug workaround
            embs = torch.nn.Parameter(embs)
        return embs

    (nparts_lhs, nparts_rhs,
     lhs_partitioned_types, rhs_partitioned_types) = get_partitioned_types(config)

    pool = create_pool(config.workers)

    model = make_model(config)

    train_config, _, _, state_dict, _ = checkpoint_manager.read_metadata()
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    model.share_memory()
    model.eval()

    for entity, econfig in config.entities.items():
        if econfig.num_partitions == 1:
            embs = load_embeddings(entity)
            model.set_embeddings(entity, embs, Side.LHS)
            model.set_embeddings(entity, embs, Side.RHS)

    all_stats: List[EvalStats] = []
    for edge_path_idx, edge_path in enumerate(config.edge_paths):
        log("Starting edge path %d / %d (%s)"
            % (edge_path_idx + 1, len(config.edge_paths), edge_path))
        edge_reader = EdgeReader(edge_path, index_base=index_base)

        all_edge_path_stats = []
        for bucket in create_buckets_ordered_lexicographically(nparts_lhs, nparts_rhs):
            tic = time.time()
            # log("%s: Loading entities" % (bucket,))

            for e in lhs_partitioned_types:
                embs = load_embeddings(e, bucket.lhs)
                model.set_embeddings(e, embs, Side.LHS)

            for e in rhs_partitioned_types:
                embs = load_embeddings(e, bucket.rhs)
                model.set_embeddings(e, embs, Side.RHS)

            # log("%s: Loading edges" % (bucket,))
            lhs, rhs, rel = edge_reader.read(bucket.lhs, bucket.rhs)
            N = rel.size(0)

            load_time = time.time() - tic
            tic = time.time()
            # log("%s: Launching and waiting for workers" % (bucket,))
            all_bucket_stats = pool.starmap(eval_one_thread, [
                (Rank(i), config, model, lhs[s], rhs[s], rel[s], evaluator)
                for i, s in enumerate(split_almost_equally(N, num_parts=config.workers))
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

            # clean up memory
            for e in lhs_partitioned_types:
                model.clear_embeddings(e)
            for e in rhs_partitioned_types:
                model.clear_embeddings(e)

        total_edge_path_stats = evaluator.sum_stats(all_edge_path_stats)
        all_stats.append(total_edge_path_stats)
        mean_edge_path_stats = total_edge_path_stats.average()
        log("")
        log("Stats for edge path %d / %d: %s"
            % (edge_path_idx + 1, len(config.edge_paths), mean_edge_path_stats))
        log("")

        yield edge_path_idx, None, mean_edge_path_stats

    mean_stats = evaluator.sum_stats(all_edge_path_stats).average()
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
