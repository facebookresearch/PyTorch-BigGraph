# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def _extract_intervals(offsets, sizes, data):
    """Select contiguous intervals of rows, given their offsets and sizes.
    E.g. suppose offsets = [35, 70, 90], sizes = [3, 2, 4], then this will
    return

            (torch.LongTensor(0, 3, 5),
            data[torch.LongTensor([35, 36, 37, 70, 71, 90, 91, 92, 93])])

    """
    offsets = offsets.long()
    sizes = sizes.long()
    res_rows = sizes.sum().item()
    assert offsets.size(0) == sizes.size(0)

    non_zero_size = sizes != 0
    if non_zero_size.long().sum() == 0:
        return torch.zeros(offsets.size(0) + 1).long(), data.new()

    new_offsets = torch.cat([torch.LongTensor([0]), sizes.cumsum(0)])
    sizes_nz = sizes[non_zero_size]
    offsets_nz = offsets[non_zero_size]

    res_delta = torch.LongTensor(res_rows).fill_(1)
    res_delta[0] = offsets_nz[0]

    if offsets_nz.size(0) > 1:
        input_delta = offsets_nz[1:] - offsets_nz[:-1] - sizes_nz[:-1]  # [32, 18]
        res_row_offsets = sizes_nz.cumsum(0)[:-1]  # [3, 5]
        res_delta[res_row_offsets] += input_delta  # [35, 1, 1, 33, 1, 19, 1, 1, 1]

    res_offsets = res_delta.cumsum(0)  # [35, 36, 37, 70, 71, 90, 91, 92, 93]
    res = data[res_offsets]

    return new_offsets, res


class TensorList(object):
    """A list of tensors of different sizes, backed by a (offset, size, data)
    tuple.

    Indexing by LongTensor returns a new TensorList with the selected list
    elements (similar to indexing a torch index_select_).

    Indexing by an int returns a torch.Tensor with that list element.
    """

    @staticmethod
    def cat(elements):
        offsets, data = zip(*[[x.offsets, x.data] for x in elements])
        offsets = list(offsets)
        batch_offset = torch.LongTensor([o[-1] for o in offsets]).cumsum(0)
        for j in range(len(offsets) - 1):
            offsets[j + 1] = offsets[j + 1][1:] + batch_offset[j]
        return TensorList(
            torch.cat(offsets),
            torch.cat(data))

    def new(self):
        return TensorList(
            torch.LongTensor([0]),
            self.data.new(),
        )

    def __init__(self, offsets, data):
        # some sanity checks
        assert(isinstance(offsets, torch.LongTensor))
        assert(offsets.ndimension() == 1)
        assert(offsets[0] == 0)
        assert(offsets[-1] == (data.size(0) if data.ndimension() > 0 else 0))

        # FIXME temporary workaround for below PyTorch bug
        # https://github.com/pytorch/pytorch/issues/5719
        if data.numel() == 0 and data.storage().size() == 0:
            storage = data.storage()
            storage.resize_(storage.size() + 1)

        self.offsets = offsets
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, torch.LongTensor):
            offsets_sub = self.offsets[index]
            sizes_sub = self.offsets[index + 1] - offsets_sub
            new_offsets, new_data = _extract_intervals(
                offsets_sub, sizes_sub, self.data)

            return TensorList(new_offsets, new_data)
        elif isinstance(index, int):
            if self.offsets[index] != self.offsets[index + 1]:
                return self.data[
                    self.offsets[index]:self.offsets[index + 1]]
            else:
                return self.data.new()
        elif isinstance(index, slice):
            assert index.step == 1 or index.step is None
            new_offsets = self.offsets[index.start:index.stop + 1]
            new_data = self.data[new_offsets[0]:new_offsets[-1]]
            new_offsets = new_offsets - new_offsets[0]
            return TensorList(new_offsets, new_data)
        else:
            raise KeyError("Unknown index type: %s" % type(index))

    def __len__(self):
        return self.offsets.size(0) - 1

    def __iadd__(self, other):
        if isinstance(other, int):
            self.data += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        if isinstance(other, int):
            self.data -= other
            return self
        else:
            raise NotImplementedError()

    def size(self, dim=None):
        # to have similar appearance with other tensor types
        assert dim == 0 or dim is None, 'TensorList can only have 1 dimension'
        if dim is None:
            return torch.Size([len(self)])
        else:
            return len(self)

    def nelement(self):
        return self.data.nelement()
