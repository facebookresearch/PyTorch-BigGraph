#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.plugin import URLPluginRegistry
from torchbiggraph.types import Partition


logger = logging.getLogger("torchbiggraph")


class AbstractEdgelistReader(ABC):

    @abstractmethod
    def __init__(self, url: str) -> None:
        pass

    @abstractmethod
    def read_edgelist(
        self,
        lhs_p: Partition,
        rhs_p: Partition,
        chunk_idx: int = 0,
        num_chunks: int = 1,
    ) -> EdgeList:
        pass


EDGELIST_READERS = URLPluginRegistry[AbstractEdgelistReader]()


# Names and values of metadata attributes for the HDF5 files.
FORMAT_VERSION_ATTR = "format_version"
FORMAT_VERSION = 1


@EDGELIST_READERS.register_as("")  # No scheme
@EDGELIST_READERS.register_as("file")
class FileEdgelistReader(AbstractEdgelistReader):
    """Reads partitioned edgelists from disk, in the format
    created by edge_downloader.py.

    Edge lists are stored as hdf5 allowing partial reads (for multi-pass).

    Currently simple implementation but should eventually be multi-threaded /
    pipelined.
    """

    def __init__(self, path: str) -> None:
        if path.startswith("file://"):
            path = path[len("file://"):]
        self.path = Path(path).resolve(strict=False)
        if not self.path.is_dir():
            raise RuntimeError(f"Invalid edge dir: {self.path}")

    def read_edgelist(
        self,
        lhs_p: Partition,
        rhs_p: Partition,
        chunk_idx: int = 0,
        num_chunks: int = 1,
    ) -> EdgeList:
        file_path = self.path / f"edges_{lhs_p}_{rhs_p}.h5"
        if not file_path.is_file():
            raise RuntimeError(f"{file_path} does not exist")
        with h5py.File(file_path, 'r') as hf:
            if hf.attrs.get(FORMAT_VERSION_ATTR, None) != FORMAT_VERSION:
                raise RuntimeError(f"Version mismatch in edge file {file_path}")
            lhs_ds = hf['lhs']
            rhs_ds = hf['rhs']
            rel_ds = hf['rel']

            num_edges = rel_ds.len()
            begin = int(chunk_idx * num_edges / num_chunks)
            end = int((chunk_idx + 1) * num_edges / num_chunks)
            chunk_size = end - begin

            lhs = torch.empty((chunk_size,), dtype=torch.long)
            rhs = torch.empty((chunk_size,), dtype=torch.long)
            rel = torch.empty((chunk_size,), dtype=torch.long)

            # Needed because https://github.com/h5py/h5py/issues/870.
            if chunk_size > 0:
                lhs_ds.read_direct(lhs.numpy(), source_sel=np.s_[begin:end])
                rhs_ds.read_direct(rhs.numpy(), source_sel=np.s_[begin:end])
                rel_ds.read_direct(rel.numpy(), source_sel=np.s_[begin:end])

            lhsd = self.read_dynamic(hf, 'lhsd', begin, end)
            rhsd = self.read_dynamic(hf, 'rhsd', begin, end)

            return EdgeList(EntityList(lhs, lhsd),
                            EntityList(rhs, rhsd),
                            rel)

    @staticmethod
    def read_dynamic(
        hf: h5py.File,
        key: str,
        begin: int,
        end: int,
    ) -> TensorList:
        try:
            offsets_ds = hf[f"{key}_offsets"]
            data_ds = hf[f"{key}_data"]
        except LookupError:
            # Empty tensor_list representation
            return TensorList(
                offsets=torch.zeros((), dtype=torch.long).expand(end - begin + 1),
                data=torch.empty((0,), dtype=torch.long))

        offsets = torch.empty((end - begin + 1,), dtype=torch.long)
        offsets_ds.read_direct(offsets.numpy(), source_sel=np.s_[begin:end + 1])
        data_begin = offsets[0].item()
        data_end = offsets[-1].item()
        data = torch.empty((data_end - data_begin,), dtype=torch.long)
        # Needed because https://github.com/h5py/h5py/issues/870.
        if data_end - data_begin > 0:
            data_ds.read_direct(data.numpy(), source_sel=np.s_[data_begin:data_end])

        offsets -= int(offsets[0])

        return TensorList(offsets, data)
