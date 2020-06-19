#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchbiggraph.model
from torchbiggraph.operators import AbstractDynamicOperator, AbstractOperator
from torchbiggraph.plugin import PluginRegistry
from torchbiggraph.types import FloatTensorType, LongTensorType


class AbstractRegularizer(ABC):
    """
    Computes a weighted penalty for embeddings involved in score computations.
    """

    def __init__(self, weight):
        self.weight = weight

    @abstractmethod
    def forward_dynamic(
        self,
        src_pos: FloatTensorType,
        dst_pos: FloatTensorType,
        src_operators: Optional[FloatTensorType],
        dst_operators: Optional[FloatTensorType],
    ) -> FloatTensorType:
        pass

    @abstractmethod
    def forward(
        self,
        src_pos: FloatTensorType,
        dst_pos: FloatTensorType,
        src_operators: Optional[FloatTensorType],
        dst_operators: Optional[FloatTensorType],
    ) -> FloatTensorType:
        pass


REGULARIZERS = PluginRegistry[AbstractRegularizer]()


@REGULARIZERS.register_as("N3")
class N3Regularizer(AbstractRegularizer):
    """N3 regularizer described in https://arxiv.org/pdf/1806.07297.pdf
    """

    def reg_embs(
        self, src_pos: FloatTensorType, dst_pos: FloatTensorType
    ) -> FloatTensorType:
        a, b, rank = torchbiggraph.model.match_shape(src_pos, -1, -1, -1)
        torchbiggraph.model.match_shape(dst_pos, a, b, rank)
        total = 0
        for x in (src_pos, dst_pos):
            total += torch.sum(self.modulus(x, rank // 2) ** 3)
        return total * self.weight

    def forward_dynamic(
        self,
        src_pos: FloatTensorType,
        dst_pos: FloatTensorType,
        operator: AbstractDynamicOperator,
        rel_idxs: LongTensorType,
    ) -> FloatTensorType:
        total = 0
        operator_params = operator.get_operator_params_for_reg(rel_idxs)
        if operator_params is not None:
            total += torch.sum(operator_params ** 3).to(src_pos.device)
        for x in (src_pos, dst_pos):
            total += torch.sum(operator.prepare_embs_for_reg(x) ** 3)
        total *= self.weight
        return total

    def forward(
        self,
        src_pos: FloatTensorType,
        dst_pos: FloatTensorType,
        operator: AbstractOperator,
    ) -> FloatTensorType:
        total = 0
        operator_params = operator.get_operator_params_for_reg()
        if operator_params is not None:
            batch_size = len(src_pos)
            total += torch.sum(operator_params ** 3) * batch_size
        for x in (src_pos, dst_pos):
            total += torch.sum(operator.prepare_embs_for_reg(x) ** 3)
        total *= self.weight
        return total
