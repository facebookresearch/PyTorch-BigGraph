#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type, TypeVar, Union, overload

import torch
from torch_extensions.tensorlist.tensorlist import TensorList


EntityListType = TypeVar('EntityListType', bound='EntityList')


class EntityList:
    """Served as a wrapper of id-based entity and featurized entity.

    self.tensor is an id-based entity list
    self.tensor_list is a featurized entity list

    This class maintains the indexing and slicing of these two parallel
    representations.
    """
    @classmethod
    def new_with_tensor(
        cls: Type[EntityListType],
        tensor: torch.FloatTensor,
    ) -> EntityListType:
        # sanity check
        assert tensor.squeeze().ndimension() == 1

        tensor = tensor.squeeze()
        tensor_list = TensorList(
            torch.zeros(tensor.nelement() + 1).long(),
            torch.Tensor([])
        )
        return cls(tensor, tensor_list)

    def __init__(
        self,
        tensor: torch.FloatTensor,
        tensor_list: TensorList,
    ) -> None:
        self.tensor: torch.FloatTensor = tensor
        self.tensor_list: TensorList = tensor_list

    # FIXME Improve typing using Literal and @overload
    def collapse(self, is_featurized: bool) -> Union[torch.FloatTensor, TensorList]:
        if is_featurized:
            return self.tensor_list
        else:
            return self.tensor

    def __getitem__(
        self: EntityListType,
        index: Union[int, slice, torch.LongTensor],
    ) -> EntityListType:
        if isinstance(index, torch.LongTensor) or isinstance(index, int):
            tensor_sub = self.tensor[index]
            tensor_list_sub = self.tensor_list[index]
            return type(self)(tensor_sub, tensor_list_sub)

        elif isinstance(index, slice):
            assert index.step == 1 or index.step is None
            tensor_sub = self.tensor[index.start:index.stop]
            tensor_list_sub = self.tensor_list[index.start:index.stop]
            return type(self)(tensor_sub, tensor_list_sub)
        else:
            raise KeyError("Unknown index type: %s" % type(index))

    def __len__(self) -> int:
        return self.tensor.nelement()

    def __iadd__(self: EntityListType, other: int) -> EntityListType:
        if isinstance(other, int):
            self.tensor += other
            self.tensor_list += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self: EntityListType, other: int) -> EntityListType:
        if isinstance(other, int):
            self.tensor -= other
            self.tensor_list -= other
            return self
        else:
            raise NotImplementedError()

    def new(self: EntityListType) -> EntityListType:
        return type(self)(self.tensor.new(), self.tensor_list.new())

    def nelement(self) -> int:
        return len(self)

    @overload
    def size(self, dim: None) -> torch.Size:
        ...

    @overload  # noqa: F811  # FIXME(T20027161)
    def size(self, dim: int) -> int:
        ...

    def size(self, dim=None):  # noqa: F811  # FIXME(T20027161)
        assert dim == 0 or dim is None, 'EntityList can only have 1 dimension'
        if dim is None:
            return torch.Size([len(self)])
        else:
            return len(self)
