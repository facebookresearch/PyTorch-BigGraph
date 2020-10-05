#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from torchbiggraph.config import ConfigSchema
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.graph_storages import EDGE_STORAGES
from torchbiggraph.losses import LOSS_FUNCTIONS
from torchbiggraph.model import Scores
from torchbiggraph.stats import Stats
from torchbiggraph.types import UNPARTITIONED


logger = logging.getLogger("torchbiggraph")


class FilteredRankingEvaluator(RankingEvaluator):
    """
    This Evaluator is meant for datasets such as FB15K, FB15K-237, WN18, WN18RR
    used in knowledge base completion. We only support one non featurized,
    non-partitioned entity type and evaluation with all negatives to be
    comparable to standard benchmarks.
    """

    def __init__(self, config: ConfigSchema, filter_paths: List[str]) -> None:
        loss_fn = LOSS_FUNCTIONS.get_class(config.loss_fn)(margin=config.margin)
        relation_weights = [r.weight for r in config.relations]
        super().__init__(loss_fn, relation_weights)

        if len(config.relations) != 1 or len(config.entities) != 1:
            raise RuntimeError(
                "Filtered ranking evaluation should only be used "
                "with dynamic relations and one entity type."
            )
        if not config.relations[0].all_negs:
            raise RuntimeError("Filtered Eval can only be done with all negatives.")
        (entity,) = config.entities.values()
        if entity.featurized:
            raise RuntimeError("Entity cannot be featurized for filtered eval.")
        if entity.num_partitions > 1:
            raise RuntimeError("Entity cannot be partitioned for filtered eval.")

        self.lhs_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.rhs_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for path in filter_paths:
            logger.info(f"Building links map from path {path}")
            e_storage = EDGE_STORAGES.make_instance(path)
            # Assume unpartitioned.
            edges = e_storage.load_edges(UNPARTITIONED, UNPARTITIONED)
            for idx in range(len(edges)):
                # Assume non-featurized.
                cur_lhs = int(edges.lhs.to_tensor()[idx])
                # Assume dynamic relations.
                cur_rel = int(edges.rel[idx])
                # Assume non-featurized.
                cur_rhs = int(edges.rhs.to_tensor()[idx])

                self.lhs_map[cur_lhs, cur_rel].append(cur_rhs)
                self.rhs_map[cur_rhs, cur_rel].append(cur_lhs)

            logger.info(f"Done building links map from path {path}")

    def _adjust_scores(self, scores: Scores, batch_edges: EdgeList):

        for idx in range(len(batch_edges)):
            # Assume non-featurized.
            cur_lhs = int(batch_edges.lhs.to_tensor()[idx])
            # Assume dynamic relations.
            cur_rel = int(batch_edges.rel[idx])
            # Assume non-featurized.
            cur_rhs = int(batch_edges.rhs.to_tensor()[idx])

            rhs_edges_filtered = self.lhs_map[cur_lhs, cur_rel]
            lhs_edges_filtered = self.rhs_map[cur_rhs, cur_rel]
            assert cur_lhs in lhs_edges_filtered
            assert cur_rhs in rhs_edges_filtered

            # The rank is computed as the number of non-negative margins (as
            # that means a negative with at least as good a score as a positive)
            # so to avoid counting positives we give them a negative margin.
            scores.lhs_neg[idx][lhs_edges_filtered] = -1e9
            scores.rhs_neg[idx][rhs_edges_filtered] = -1e9
