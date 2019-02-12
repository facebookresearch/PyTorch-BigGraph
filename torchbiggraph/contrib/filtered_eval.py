#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Union, List

import torch

from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.config import ConfigSchema
from torchbiggraph.eval import RankingEvaluator
from torchbiggraph.fileio import EdgeReader
from torchbiggraph.model import Margins, Scores
from torchbiggraph.stats import Stats
from torchbiggraph.util import log, infer_input_index_base, Partition


class FilteredRankingEvaluator(RankingEvaluator):
    """
    This Evaluator is meant for dataset such as FB15K, FB15K-237, WN18, WN18RR used in knowledge base completion
    we only support one non featurized, non-partitioned entity type and
    evaluation with all negatives to be comparable to standard benchmarks.
    """
    def __init__(self, config: ConfigSchema, filter_paths : List[str]):
        if len(config.relations) != 1 or len(config.entities) != 1:
            raise RuntimeError("Filtered Eval should only be used with dynamic relations and one entity type.")
        if not config.relations[0].all_negs:
            raise RuntimeError("Filtered Eval can only be done with all negatives.")

        entity, = config.entities.values()
        if entity.featurized:
            raise RuntimeError("Entity cannot be featurized for filtered eval.")
        if entity.num_partitions > 1:
            raise RuntimeError("Entity cannot be partitioned for filtered eval.")

        index_base = infer_input_index_base(config)
        self.lhs_map = defaultdict(list)
        self.rhs_map = defaultdict(list)
        for path in filter_paths:
            log("Building links map from path " + path)
            e_reader = EdgeReader(path, index_base=index_base)
            lhs, rhs, rel = e_reader.read(Partition(0), Partition(0))  # assume non-partitioned
            N = lhs.size(0)
            for i in range(N):
                cur_lhs = lhs[i].collapse(is_featurized=False).item()  # assume non-featurized
                cur_rel = rel[i].item()
                cur_rhs = rhs[i].collapse(is_featurized=False).item()  # assume non-featurized

                self.lhs_map[cur_lhs, cur_rel].append(cur_rhs)
                self.rhs_map[cur_rhs, cur_rel].append(cur_lhs)

            log("Done building links map from path " + path)

    def eval(
        self,
        scores: Scores,
        margins: Margins,
        batch_lhs: Union[torch.FloatTensor, TensorList],
        batch_rhs: Union[torch.FloatTensor, TensorList],
        batch_rel: Union[int, torch.LongTensor],
    ) -> Stats:
        b = batch_lhs.size(0)
        for idx in range(0, b):
            cur_lhs = batch_lhs[idx].item()
            cur_rel = batch_rel[idx].item()
            cur_rhs = batch_rhs[idx].item()
            rhs_edges_filtered = self.lhs_map[cur_lhs, cur_rel]
            lhs_edges_filtered = self.rhs_map[cur_rhs, cur_rel]

            # Since NonFilteredEvaluator computes the rank as sum(margin >=0)
            # we make the margin of positive entities < 0 to ignore them in the rank computation
            margins[0][idx][lhs_edges_filtered] = -1
            margins[1][idx][rhs_edges_filtered] = -1
            assert cur_lhs in lhs_edges_filtered
            assert cur_rhs in rhs_edges_filtered

        return super().eval(scores, margins, batch_lhs, batch_rhs, batch_rel)
