#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchbiggraph.model import match_shape
from torchbiggraph.plugin import PluginRegistry
from torchbiggraph.types import FloatTensorType


class AbstractLossFunction(nn.Module, ABC):

    """Calculate weighted loss of scores for positive and negative pairs.

    The inputs are a 1-D tensor of size P containing scores for positive pairs
    of entities (i.e., those among which an edge exists) and a P x N tensor
    containing scores for negative pairs (i.e., where no edge should exist). The
    pairs of entities corresponding to pos_scores[i] and to neg_scores[i,j] have
    at least one endpoint in common. The output is the loss value these scores
    induce. If the method supports weighting (as is the case for the logistic
    loss) all positive scores will be weighted by the same weight and so will
    all the negative ones.
    """

    @abstractmethod
    def forward(
        self,
        pos_scores: FloatTensorType,
        neg_scores: FloatTensorType,
    ) -> FloatTensorType:
        pass


LOSS_FUNCTIONS = PluginRegistry[AbstractLossFunction]()


@LOSS_FUNCTIONS.register_as("logistic")
class LogisticLossFunction(AbstractLossFunction):

    def forward(
        self,
        pos_scores: FloatTensorType,
        neg_scores: FloatTensorType,
    ) -> FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)
        neg_weight = 1 / num_neg if num_neg > 0 else 0

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.ones(()).expand(num_pos),
            reduction='sum',
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.zeros(()).expand(num_pos, num_neg),
            reduction='sum',
        )
        loss = pos_loss + neg_weight * neg_loss

        return loss


@LOSS_FUNCTIONS.register_as("ranking")
class RankingLossFunction(AbstractLossFunction):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: FloatTensorType,
        neg_scores: FloatTensorType,
    ) -> FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)

        # FIXME Workaround for https://github.com/pytorch/pytorch/issues/15223.
        if num_pos == 0 or num_neg == 0:
            return torch.zeros((), requires_grad=True)

        loss = F.margin_ranking_loss(
            neg_scores,
            pos_scores.unsqueeze(1),
            target=torch.full((1, 1), -1, dtype=torch.float),
            margin=self.margin,
            reduction='sum',
        )

        return loss


@LOSS_FUNCTIONS.register_as("softmax")
class SoftmaxLossFunction(AbstractLossFunction):

    def forward(
        self,
        pos_scores: FloatTensorType,
        neg_scores: FloatTensorType,
    ) -> FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)

        # FIXME Workaround for https://github.com/pytorch/pytorch/issues/15870
        # and https://github.com/pytorch/pytorch/issues/15223.
        if num_pos == 0 or num_neg == 0:
            return torch.zeros((), requires_grad=True)

        scores = torch.cat([
            pos_scores.unsqueeze(1), neg_scores.logsumexp(dim=1, keepdim=True)
        ], dim=1)
        loss = F.cross_entropy(
            scores,
            torch.zeros((), dtype=torch.long).expand(num_pos),
            reduction='sum',
        )

        return loss
