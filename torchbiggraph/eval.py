#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from itertools import chain
from typing import Generator, Optional, Tuple

import attr
import torch

from .config import parse_config, ConfigSchema
from .fileio import CheckpointManager, EdgeReader
from .model import RankingLoss, make_model, override_model
from .util import log, get_partitioned_types, chunk_by_index, create_workers, \
    join_workers, update_config_for_dynamic_relations, \
    compute_randomized_auc, Side
from .stats import Stats, stats


@stats
class EvalStats(Stats):
    pos_rank: float = attr.ib()
    mrr: float = attr.ib()
    r1: float = attr.ib()
    r10: float = attr.ib()
    r50: float = attr.ib()
    auc: float = attr.ib()


def eval_one_batch(model, batch_lhs, batch_rhs, batch_rel):
    B = batch_lhs.size(0)
    batch_lhs = batch_lhs.collapse(model.is_featurized(batch_rel, Side.LHS))
    batch_rhs = batch_rhs.collapse(model.is_featurized(batch_rel, Side.RHS))
    # For evaluation, we want the ranking, not margin loss, so set
    # loss_fn = "ranking"
    # margin = 0
    with override_model(model, loss_fn=RankingLoss(0)):
        loss, (lhs_margin, rhs_margin), scores = model(
            batch_lhs,
            batch_rhs,
            batch_rel)

    # lhs_margin[i,j] is the score of (neg_lhs[i,j], rel[i], rhs[i]) minus the
    # score of (lhs[i], rel[i], rhs[i]). Thus it is >= 0 when the i-th positive
    # edge achieved a lower score than the j-th negative that we sampled for it.
    lrank = lhs_margin.ge(0).float().sum(1) + 1
    rrank = rhs_margin.ge(0).float().sum(1) + 1

    auc = .5 * compute_randomized_auc(scores[0], scores[2], B)\
          + .5 * compute_randomized_auc(scores[1], scores[3], B)

    # it's kind of weird that we average violators from LHS and RHS
    # but I'm just going to copy the old version for now

    return EvalStats(
        pos_rank=(lrank.float().sum().item() + rrank.float().sum().item()) / 2,
        mrr=(lrank.float().reciprocal().sum().item()
             + rrank.float().reciprocal().sum().item()) / 2,
        r1=(lrank.le(1).float() + rrank.le(1).float()).sum().item() / 2,
        r10=(lrank.le(10).float() + rrank.le(10).float()).sum().item() / 2,
        r50=(lrank.le(50).float() + rrank.le(50).float()).sum().item() / 2,
        auc=auc * B,  # at the end the auc will be averaged over count
        count=B)


def eval_many_batches(config, model, lhs, rhs, rel):
    all_stats = []
    if model.num_dynamic_rels > 0:
        offset, N = 0, rel.size(0)
        while offset < N:
            B = min(N - offset, config.batch_size)
            all_stats.append(eval_one_batch(
                model,
                lhs[offset:offset + B],
                rhs[offset:offset + B],
                rel[offset:offset + B]))
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
                all_stats.append(eval_one_batch(
                    model, batch_lhs, batch_rhs, rel_type))

    return EvalStats.sum(all_stats)


def eval_one_thread(rank, config, model, lhs, rhs, rel):
    """ This is the eval loop executed by each HOGWILD thread.
    """
    stats = eval_many_batches(config, model, lhs, rhs, rel)
    print("Rank %d done" % rank)
    return stats


def do_eval_and_report_stats(
    config: ConfigSchema,
) -> Generator[Tuple[int, Optional[Tuple[int, int]], EvalStats], None, None]:
    """Computes eval metrics (r1/r10/r50) for a checkpoint with trained
       embeddings.
    """

    config = update_config_for_dynamic_relations(config)

    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    def load_embeddings(entity, part=0):
        data = checkpoint_manager.read(entity, part, strict=True)
        embs, _optim_state = data
        embs.share_memory_()
        if not isinstance(embs, torch.nn.Parameter):  # Pytorch bug workaround
            embs = torch.nn.Parameter(embs)
        return embs

    (nparts_lhs, nparts_rhs,
     lhs_partitioned_types, rhs_partitioned_types) = get_partitioned_types(config)

    processes, qIn, qOut = create_workers(config.workers, eval_one_thread, config)

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

    for epoch in range(len(config.edge_paths)):  # FIXME: maybe just one epoch?
        edgePath = config.edge_paths[epoch]
        log("Starting epoch %d / %d; edgePath= %s" %
            (epoch + 1, len(config.edge_paths), config.edge_paths[epoch]))
        edge_reader = EdgeReader(edgePath)

        all_epoch_stats = []
        for lhsP in range(nparts_lhs):
            for rhsP in range(nparts_rhs):
                tic = time.time()
                # log("( %d , %d ): Loading entities" % (lhsP, rhsP))

                for e in lhs_partitioned_types:
                    embs = load_embeddings(e, lhsP)
                    model.set_embeddings(e, embs, Side.LHS)

                for e in rhs_partitioned_types:
                    embs = load_embeddings(e, rhsP)
                    model.set_embeddings(e, embs, Side.RHS)

                # log("( %d , %d ): Loading edges" % (lhsP, rhsP))
                lhs, rhs, rel = edge_reader.read(lhsP, rhsP)
                N = rel.size(0)

                load_time = time.time() - tic
                tic = time.time()
                # log("( %d , %d ): Launching workers" % (lhsP, rhsP))
                for rank in range(config.workers):
                    start = int(rank * N / config.workers)
                    end = int((rank + 1) * N / config.workers)
                    qIn[rank].put(
                        (model, lhs[start:end], rhs[start:end], rel[start:end]))

                # log("( %d , %d ): Waiting for workers" % (lhsP, rhsP))
                all_stats = []
                for rank in range(config.workers):
                    all_stats.append(qOut[rank].get())

                compute_time = time.time() - tic
                log("( %d , %d ): Processed %d edges in %.2g s (%.2gM/sec);"
                    " load time: %.2g s" %
                    (lhsP, rhsP, lhs.size(0), compute_time,
                     lhs.size(0) / compute_time / 1e6, load_time))

                total_stats = EvalStats.sum(all_stats)
                all_epoch_stats.append(total_stats)
                mean_stats = total_stats.average()

                log("( %d , %d ): %s" % (lhsP, rhsP, mean_stats))

                yield epoch, (lhsP, rhsP), mean_stats

                # clean up memory
                for e in lhs_partitioned_types:
                    model.clear_embeddings(e)
                for e in rhs_partitioned_types:
                    model.clear_embeddings(e)

        mean_epoch_stats = EvalStats.sum(all_epoch_stats).average()
        log("")
        log("Epoch %d full stats: %s" % (epoch + 1, mean_epoch_stats))
        log("")

        yield epoch, None, mean_epoch_stats

    join_workers(processes, qIn, qOut)


def do_eval(
    config: ConfigSchema,
) -> None:
    # Create and run the generator until exhaustion.
    for _ in do_eval_and_report_stats(config):
        pass


def main():
    torch.set_num_threads(1)

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
    # log(config)
    do_eval(config)


if __name__ == '__main__':
    main()
