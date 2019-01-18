#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensorlist.tensorlist import TensorList


class EntityList(object):
    """Served as a wrapper of id-based entity and featurized entity.

    self.tensor is an id-based entity list
    self.tensor_list is a featurized entity list

    This class maintains the indexing and slicing of these two parallel
    representations.
    """
    @staticmethod
    def newWithTensor(tensor):
        # sanity check
        assert tensor.squeeze().ndimension() == 1

        tensor = tensor.squeeze()
        tensor_list = TensorList(
            torch.zeros(tensor.nelement() + 1).long(),
            torch.Tensor([])
        )
        return EntityList(tensor, tensor_list)

    def __init__(self, tensor, tensor_list):
        self.tensor = tensor
        self.tensor_list = tensor_list

    def collapse(self, is_featurized):
        if is_featurized:
            return self.tensor_list
        else:
            return self.tensor

    def __getitem__(self, index):
        if isinstance(index, torch.LongTensor) or isinstance(index, int):
            tensor_sub = self.tensor[index]
            tensor_list_sub = self.tensor_list[index]
            return EntityList(tensor_sub, tensor_list_sub)

        elif isinstance(index, slice):
            assert index.step == 1 or index.step is None
            tensor_sub = self.tensor[index.start:index.stop]
            tensor_list_sub = self.tensor_list[index.start:index.stop]
            return EntityList(tensor_sub, tensor_list_sub)
        else:
            raise KeyError("Unknown index type: %s" % type(index))

    def __len__(self):
        return self.tensor.nelement()

    def __iadd__(self, other):
        if isinstance(other, int):
            self.tensor += other
            self.tensor_list += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        if isinstance(other, int):
            self.tensor -= other
            self.tensor_list -= other
            return self
        else:
            raise NotImplementedError()

    def new(self):
        return EntityList(self.tensor.new(), self.tensor_list.new())

    def nelement(self):
        return len(self)

    def size(self, dim=None):
        assert dim == 0 or dim is None, 'EntityList can only have 1 dimension'
        if dim is None:
            return torch.Size([len(self)])
        else:
            return len(self)
