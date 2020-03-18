#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from typing import Any, Sequence, Union

import torch
from torchbiggraph.tensorlist import TensorList
from torchbiggraph.types import LongTensorType


class EntityList:
    """Served as a wrapper of id-based entity and featurized entity.

    self.tensor is an id-based entity list
    self.tensor_list is a featurized entity list

    This class maintains the indexing and slicing of these two parallel
    representations.
    """

    @classmethod
    def empty(cls) -> "EntityList":
        return cls(torch.empty((0,), dtype=torch.long), TensorList.empty())

    @classmethod
    def from_tensor(cls, tensor: LongTensorType) -> "EntityList":
        if tensor.dim() != 1:
            raise ValueError("Expected 1D tensor, got %dD" % tensor.dim())
        tensor_list = TensorList.empty(num_tensors=tensor.shape[0])
        return cls(tensor, tensor_list)

    @classmethod
    def from_tensor_list(cls, tensor_list: TensorList) -> "EntityList":
        tensor = torch.full((len(tensor_list),), -1, dtype=torch.long)
        return cls(tensor, tensor_list)

    @classmethod
    def cat(cls, entity_lists: Sequence["EntityList"]) -> "EntityList":
        return cls(
            torch.cat([el.tensor for el in entity_lists]),
            TensorList.cat(el.tensor_list for el in entity_lists),
        )

    def __init__(self, tensor: LongTensorType, tensor_list: TensorList) -> None:
        if not isinstance(tensor, (torch.LongTensor, torch.cuda.LongTensor)):
            raise TypeError(
                "Expected long tensor as first argument, got %s" % type(tensor)
            )
        if not isinstance(tensor_list, TensorList):
            raise TypeError(
                "Expected tensor list as second argument, got %s" % type(tensor_list)
            )
        if tensor.dim() != 1:
            raise ValueError(
                "Expected 1-dimensional tensor, got %d-dimensional one" % tensor.dim()
            )
        if tensor.shape[0] != len(tensor_list):
            raise ValueError(
                "The tensor and tensor list have different lengths: %d != %d"
                % (tensor.shape[0], len(tensor_list))
            )
        # TODO We could check that, for all i, we have either tensor[i] < 0 or
        # tensor_list[i] empty, however it's expensive and we're already doing
        # something similar at retrieval inside to_tensor(_list).
        self.tensor: LongTensorType = tensor
        self.tensor_list: TensorList = tensor_list

    def to_tensor(self) -> LongTensorType:
        if len(self.tensor_list.data) != 0:
            raise RuntimeError(
                "Getting the tensor data of an EntityList "
                "that also has tensor list data"
            )
        return self.tensor

    def to_tensor_list(self) -> TensorList:
        if not self.tensor.eq(-1).all():
            raise RuntimeError(
                "Getting the tensor list data of an EntityList "
                "that also has tensor data"
            )
        return self.tensor_list

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EntityList):
            return NotImplemented
        return (
            torch.equal(self.tensor, other.tensor)
            and torch.equal(self.tensor_list.offsets, other.tensor_list.offsets)
            and torch.equal(self.tensor_list.data, other.tensor_list.data)
        )

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return "EntityList(%r, TensorList(%r, %r))" % (
            self.tensor,
            self.tensor_list.offsets,
            self.tensor_list.data,
        )

    def __getitem__(self, index: Union[int, slice, LongTensorType]) -> "EntityList":
        if isinstance(index, int):
            return self[index : index + 1]

        if isinstance(index, (torch.LongTensor, torch.cuda.LongTensor)) or isinstance(
            index, int
        ):
            tensor_sub = self.tensor[index]
            tensor_list_sub = self.tensor_list[index]
            return type(self)(tensor_sub, tensor_list_sub)

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Expected slice with step 1, got %d" % step)
            tensor_sub = self.tensor[start:stop]
            tensor_list_sub = self.tensor_list[start:stop]
            return type(self)(tensor_sub, tensor_list_sub)

        raise KeyError("Unknown index type: %s" % type(index))

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def to(self, *args, **kwargs) -> "EntityList":
        return type(self)(
            self.tensor.to(*args, **kwargs), self.tensor_list.to(*args, **kwargs)
        )
