#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import errno
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from torchbiggraph.types import (
    FloatTensorType,
    ModuleStateDict,
    OptimizerStateDict,
)


logger = logging.getLogger("torchbiggraph")


NP_VOID_DTYPE = np.dtype("V1")


class DatasetIO(io.RawIOBase):
    """A file-like proxy to a HDF5 dataset

    Given a one-dimensional HFD5 dataset object whose elements are bytes, this
    class wraps it and provides access to it through a file-like interface. The
    "file" is open in binary mode (i.e. returns bytes objects rather than strs),
    is read-only (writing could be easily supported, but isn't needed), seekable
    and only offers "raw" (unbuffered) I/O. Users will probably want to wrap it
    in a BufferedReader for better performance.

    This is needed as a compatibility layer to enable features that only support
    file-like objects (like torch.load) to read from HDF5-backed storage and
    only load data as-needed (rather than pre-loading everything, as would be
    necessary with BytesIO).

    Writing isn't supported because (non-chunked) HDF5 datasets must be created
    with their final size known in advance, which is usually not possible with
    a file-like interface.
    """

    def __init__(self, dataset: h5py.Dataset):
        if dataset.dtype != NP_VOID_DTYPE:
            raise TypeError("Dataset doesn't contain bytes")
        if dataset.shape != (dataset.size,):
            raise TypeError("Dataset isn't a one-dimensional array")
        self.dataset = dataset
        self.pos = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        array = np.frombuffer(buffer, dtype=NP_VOID_DTYPE)
        size = min(len(buffer), self.dataset.size - self.pos)
        # Needed because https://github.com/h5py/h5py/issues/870.
        if size > 0:
            self.dataset.read_direct(array, np.s_[self.pos:self.pos + size], np.s_[:size])
        self.pos += size
        return size

    def readall(self) -> bytes:
        # We're supposed to implement this, but it doesn't appear to be needed.
        raise io.UnsupportedOperation()

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence is io.SEEK_SET:
            self.pos = offset
        if whence is io.SEEK_CUR:
            self.pos += offset
        if whence is io.SEEK_END:
            self.pos = self.dataset.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos


# Names and values of metadata attributes for the HDF5 files.
FORMAT_VERSION_ATTR = "format_version"
FORMAT_VERSION = 1
STATE_DICT_KEY_ATTR = "state_dict_key"
# Names of groups and datasets inside the HDF5 files.
EMBEDDING_DATASET = "embeddings"
MODEL_STATE_DICT_GROUP = "model"
OPTIMIZER_STATE_DICT_DATASET = "optimizer/state_dict"


def save_embeddings(hf: h5py.File, embeddings: FloatTensorType) -> None:
    hf.create_dataset(EMBEDDING_DATASET, data=embeddings.numpy())


def load_embeddings(hf: h5py.File) -> FloatTensorType:
    dataset: h5py.Dataset = hf[EMBEDDING_DATASET]
    storage = torch.FloatStorage._new_shared(dataset.size)
    embeddings = torch.FloatTensor(storage).view(dataset.shape)
    # Needed because https://github.com/h5py/h5py/issues/870.
    if dataset.size > 0:
        dataset.read_direct(embeddings.numpy())
    return embeddings


def save_optimizer_state_dict(
    hf: h5py.File,
    state_dict: Optional[OptimizerStateDict],
) -> None:
    if state_dict is None:
        return
    with io.BytesIO() as fobj:
        torch.save(state_dict, fobj)
        hf.create_dataset(OPTIMIZER_STATE_DICT_DATASET,
                          data=np.frombuffer(fobj.getbuffer(), dtype=NP_VOID_DTYPE))


def load_optimizer_state_dict(hf: h5py.File) -> Optional[OptimizerStateDict]:
    if OPTIMIZER_STATE_DICT_DATASET not in hf:
        return None
    with io.BufferedReader(DatasetIO(hf[OPTIMIZER_STATE_DICT_DATASET])) as fobj:
        return torch.load(fobj)


class OneWayMapping:

    def __init__(self, src: str, dst: str, fields: List[str]) -> None:
        self.src = re.compile(src.format(**{f: r"(?P<%s>[^./]+)" % f for f in fields}))
        self.dst = dst.format(**{f: r"\g<%s>" % f for f in fields})

    def map(self, name: str) -> str:
        match = self.src.fullmatch(name)
        if match is None:
            raise ValueError()
        return match.expand(self.dst)


class TwoWayMapping:

    def __init__(self, private: str, public: str, fields: List[str]) -> None:
        self.private_to_public = OneWayMapping(private.replace(".", r"\."), public, fields)
        self.public_to_private = OneWayMapping(public, private, fields)


def save_model_state_dict(
    hf: h5py.File,
    state_dict: ModuleStateDict,
    mappings: List[TwoWayMapping],
) -> None:
    g = hf.create_group(MODEL_STATE_DICT_GROUP, track_order=True)
    for private_key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Isn't the state dict supposed to be "
                               "a shallow key-to-tensor mapping?!")
        for mapping in mappings:
            try:
                public_key = mapping.private_to_public.map(private_key)
            except ValueError:
                continue
            else:
                break
        else:
            raise RuntimeError("Couldn't find a match for state dict key: %s"
                               % private_key)

        dataset = g.create_dataset(public_key, data=tensor.numpy())
        dataset.attrs[STATE_DICT_KEY_ATTR] = private_key


def load_model_state_dict(
    hf: h5py.File,
    mappings: List[TwoWayMapping],
) -> Optional[ModuleStateDict]:
    if MODEL_STATE_DICT_GROUP not in hf:
        return None
    g = hf[MODEL_STATE_DICT_GROUP]
    state_dict = ModuleStateDict({})

    def process_dataset(public_key, dataset) -> None:
        if not isinstance(dataset, h5py.Dataset):
            return
        for mapping in mappings:
            try:
                private_key = mapping.public_to_private.map(public_key)
            except ValueError:
                continue
            else:
                break
        else:
            raise RuntimeError("Couldn't find a match for dataset name: %s"
                               % public_key)
        state_dict[private_key] = torch.from_numpy(dataset[...])

    g.visititems(process_dataset)
    return state_dict


def save_entity_partition(
    path: str,
    embs: FloatTensorType,
    optim_state: Optional[OptimizerStateDict],
    metadata: Dict[str, Any],
) -> None:
    logger.debug(f"Saving to {path}")
    with h5py.File(path, "w") as hf:
        hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
        for k, v in metadata.items():
            hf.attrs[k] = v
        save_embeddings(hf, embs)
        save_optimizer_state_dict(hf, optim_state)
        hf.flush()
    logger.debug(f"Done saving to {path}")


def save_model(
    path: str,
    state_dict: ModuleStateDict,
    optim_state: Optional[OptimizerStateDict],
    metadata: Dict[str, Any],
    mappings: List[TwoWayMapping],
) -> None:
    logger.debug(f"Saving to {path}")
    with h5py.File(path, "w") as hf:
        hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
        for k, v in metadata.items():
            hf.attrs[k] = v
        save_model_state_dict(hf, state_dict, mappings)
        save_optimizer_state_dict(hf, optim_state)
        hf.flush()
    logger.debug(f"Done saving to {path}")


def load_entity_partition(
    path: str,
) -> Tuple[FloatTensorType, Optional[OptimizerStateDict]]:
    logger.debug(f"Loading from {path}")
    try:
        with h5py.File(path, "r") as hf:
            if hf.attrs.get(FORMAT_VERSION_ATTR, None) != FORMAT_VERSION:
                raise RuntimeError(f"Version mismatch in embeddings file {path}")
            embs = load_embeddings(hf)
            optim_state = load_optimizer_state_dict(hf)
    except OSError as err:
        # h5py refuses to make it easy to figure out what went wrong. The errno
        # attribute is set to None. See https://github.com/h5py/h5py/issues/493.
        if "errno = %d" % errno.ENOENT in str(err):
            raise FileNotFoundError() from err
        raise err
    logger.debug(f"Done loading from {path}")
    return embs, optim_state


def load_model(
    path: str,
    mappings: List[TwoWayMapping],
) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
    logger.debug(f"Loading from {path}")
    try:
        with h5py.File(path, "r") as hf:
            if hf.attrs.get(FORMAT_VERSION_ATTR, None) != FORMAT_VERSION:
                raise RuntimeError(f"Version mismatch in model file {path}")
            state_dict = load_model_state_dict(hf, mappings)
            optim_state = load_optimizer_state_dict(hf)
    except OSError as err:
        # h5py refuses to make it easy to figure out what went wrong. The errno
        # attribute is set to None. See https://github.com/h5py/h5py/issues/493.
        if "errno = %d" % errno.ENOENT in str(err):
            raise FileNotFoundError() from err
        raise err
    logger.debug(f"Done loading from {path}")
    return state_dict, optim_state
