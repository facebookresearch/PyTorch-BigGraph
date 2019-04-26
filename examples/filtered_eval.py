#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Union, Tuple

import torch

from torchbiggraph.config import ConfigSchema
from torchbiggraph.entitylist import EntityList
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.fileio import EdgeReader
from torchbiggraph.model import Scores
from torchbiggraph.util import log
from torchbiggraph.stats import Stats
from torchbiggraph.types import Partition, LongTensorType


class FilteredRankingEvaluator(RankingEvaluator):
    """
    This Evaluator is meant for datasets such as FB15K, FB15K-237, WN18, WN18RR
    used in knowledge base completion. We only support one non featurized,
    non-partitioned entity type and evaluation with all negatives to be
    comparable to standard benchmarks.
    """

    def __init__(self, config: ConfigSchema, filter_paths: List[str]):
        super().__init__()
        if len(config.relations) != 1 or len(config.entities) != 1:
            raise RuntimeError("Filtered ranking evaluation should only be used "
                               "with dynamic relations and one entity type.")
        if not config.relations[0].all_negs:
            raise RuntimeError("Filtered Eval can only be done with all negatives.")

        entity, = config.entities.values()
        if entity.featurized:
            raise RuntimeError("Entity cannot be featurized for filtered eval.")
        if entity.num_partitions > 1:
            raise RuntimeError("Entity cannot be partitioned for filtered eval.")

        self.lhs_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.rhs_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for path in filter_paths:
            log("Building links map from path %s" % path)
            e_reader = EdgeReader(path)
            # Assume unpartitioned.
            lhs, rhs, rel = e_reader.read(Partition(0), Partition(0))
            num_edges = lhs.size(0)
            for i in range(num_edges):
                # Assume non-featurized.
                cur_lhs = lhs.to_tensor()[i].item()
                cur_rel = rel[i].item()
                # Assume non-featurized.
                cur_rhs = rhs.to_tensor()[i].item()

                self.lhs_map[cur_lhs, cur_rel].append(cur_rhs)
                self.rhs_map[cur_rhs, cur_rel].append(cur_lhs)

            log("Done building links map from path %s" % path)

    def eval(
        self,
        scores: Scores,
        batch_lhs: EntityList,
        batch_rhs: EntityList,
        batch_rel: Union[int, LongTensorType],
    ) -> Stats:
        # Assume dynamic relations.
        assert isinstance(batch_rel, torch.LongTensor)

        _, _, lhs_neg_scores, rhs_neg_scores = scores
        b = batch_lhs.size(0)
        for idx in range(b):
            # Assume non-featurized.
            cur_lhs = batch_lhs.to_tensor()[idx].item()
            cur_rel = batch_rel[idx].item()
            # Assume non-featurized.
            cur_rhs = batch_rhs.to_tensor()[idx].item()

            rhs_edges_filtered = self.lhs_map[cur_lhs, cur_rel]
            lhs_edges_filtered = self.rhs_map[cur_rhs, cur_rel]
            assert cur_lhs in lhs_edges_filtered
            assert cur_rhs in rhs_edges_filtered

            # The rank is computed as the number of non-negative margins (as
            # that means a negative with at least as good a score as a positive)
            # so to avoid counting positives we give them a negative margin.
            lhs_neg_scores[idx][lhs_edges_filtered] = -1e9
            rhs_neg_scores[idx][rhs_edges_filtered] = -1e9

        return super().eval(scores, batch_lhs, batch_rhs, batch_rel)
