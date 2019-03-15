#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, NamedTuple, NewType, TypeVar

import torch


# torch.FloatTensor and torch.LongTensor are defined as empty subclasses of
# torch.Tensor by PyTorch's type stub, which means that any operation on them
# returns plain untyped torch.Tensors. This makes it impossible to use the typed
# subtypes to annotate functions as they wouldn't get the type they expect.
# Thus for type checking to work functions must be annotated with torch.Tensor.
# To preserve and expose that information, at least to humans, we use more
# informative aliases for torch.Tensor. (PS: FloatTensor and LongTensor are in
# fact instances of the torch.tensortype metaclass).
CharTensorType = torch.Tensor
FloatTensorType = torch.Tensor
LongTensorType = torch.Tensor


T = TypeVar("T")


class Side(Enum):
    LHS = 0
    RHS = 1

    def pick(self, lhs: T, rhs: T) -> T:
        if self is Side.LHS:
            return lhs
        elif self is Side.RHS:
            return rhs
        else:
            raise NotImplementedError("Unknown side: %s" % self)


EntityName = NewType("EntityName", str)
Rank = NewType("Rank", int)
Partition = NewType("Partition", int)
ModuleStateDict = NewType("ModuleStateDict", Dict[str, torch.Tensor])
OptimizerStateDict = NewType("OptimizerStateDict", Dict[str, Any])


class Bucket(NamedTuple):
    lhs: Partition
    rhs: Partition

    def get_partition(self, side: Side) -> Partition:
        return side.pick(self.lhs, self.rhs)

    def __str__(self) -> str:
        return "( %d , %d )" % (self.lhs, self.rhs)
