#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import errno
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Type

import h5py
import numpy as np
import torch

from torchbiggraph.plugin import URLPluginRegistry
from torchbiggraph.types import (
    EntityName,
    FloatTensorType,
    ModuleStateDict,
    Partition,
)


logger = logging.getLogger("torchbiggraph")


class CouldNotLoadData(Exception):
    pass


class ModelParameter(NamedTuple):
    # This is the "internal" name, the one of the model's state dict, which is
    # considered an implementation detail. Thus the parameters are stored under
    # different, "public", names but the original ones are still attached, as
    # attributes, to help with debugging.
    private_name: str
    tensor: torch.Tensor


class AbstractCheckpointStorage(ABC):

    @abstractmethod
    def __init__(self, url: str) -> None:
        pass

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def save_version(self, version: int) -> None:
        pass

    @abstractmethod
    def load_version(self) -> int:
        pass

    @abstractmethod
    def save_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
        embeddings: FloatTensorType,
        optim_state: Optional[bytes],
        metadata: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def load_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
    ) -> Tuple[FloatTensorType, Optional[bytes]]:
        pass

    @abstractmethod
    def drop_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
    ) -> None:
        pass

    @abstractmethod
    def save_model(
        self,
        version: int,
        state_dict: Dict[str, ModelParameter],
        optim_state: Optional[bytes],
        metadata: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def load_model(
        self,
        version: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[bytes]]:
        pass

    @abstractmethod
    def drop_model(self, version: int) -> None:
        pass

    @abstractmethod
    def save_config(self, config_json: str) -> None:
        pass

    @abstractmethod
    def load_config(self) -> str:
        pass

    @abstractmethod
    def prepare_snapshot(self, version: int, epoch_idx: int) -> None:
        pass

    @abstractmethod
    def copy_entity_partition_to_snapshot(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
        epoch_idx: int,
    ) -> None:
        pass

    @abstractmethod
    def copy_model_to_snapshot(self, version: int, epoch_idx: int) -> None:
        pass

    @abstractmethod
    def copy_version_to_snapshot(self, version: int, epoch_idx: int) -> None:
        pass


CHECKPOINT_STORAGES = URLPluginRegistry[AbstractCheckpointStorage]()


NP_VOID_DTYPE = np.dtype("V1")


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
    state_dict: Optional[bytes],
) -> None:
    if state_dict is None:
        return
    hf.create_dataset(OPTIMIZER_STATE_DICT_DATASET,
                      data=np.frombuffer(state_dict, dtype=NP_VOID_DTYPE))


def load_optimizer_state_dict(hf: h5py.File) -> Optional[bytes]:
    if OPTIMIZER_STATE_DICT_DATASET not in hf:
        return None
    return hf[OPTIMIZER_STATE_DICT_DATASET][...].tobytes()


def save_model_state_dict(
    hf: h5py.File,
    state_dict: Dict[str, ModelParameter],
) -> None:
    g = hf.create_group(MODEL_STATE_DICT_GROUP, track_order=True)
    for public_name, param in state_dict.items():
        dataset = g.create_dataset(public_name, data=param.tensor.numpy())
        dataset.attrs[STATE_DICT_KEY_ATTR] = param.private_name


def load_model_state_dict(
    hf: h5py.File,
) -> Optional[ModuleStateDict]:
    if MODEL_STATE_DICT_GROUP not in hf:
        return None
    g = hf[MODEL_STATE_DICT_GROUP]
    state_dict = ModuleStateDict({})

    def process_dataset(public_name, dataset) -> None:
        if not isinstance(dataset, h5py.Dataset):
            return
        state_dict[public_name] = torch.from_numpy(dataset[...])

    g.visititems(process_dataset)
    return state_dict


@CHECKPOINT_STORAGES.register_as("")  # No scheme
@CHECKPOINT_STORAGES.register_as("file")
class FileCheckpointStorage(AbstractCheckpointStorage):

    """Reads and writes checkpoint data to/from disk.

    Checkpoints are saved as HDF5 files. The embeddings for an entity partition
    are stored in the `embeddings_<entity>_<partition>.v<version>.h5` file.

        hf = h5py.File("embeddings_foo_0.v123.h5", "r")
        embedding_of_entity_42 = hf["embeddings"][42, :]

    The parameters that are not specific to a certain entity (i.e., all but the
    embeddings) are stored in a `model.v<version>.h5` file.

        hf = h5py.File("model.v123.h5", "r")
        keys = []
        hf["model"].visit(keys.append)
        print(keys)

    Both files also contain the state dictionary of their optimizer, and some
    metadata as attributes on the root node.

        print(list(hf.attrs))

    Swapped-out partitions are saved to disk with an incremented version number.
    Once a training iteration completes, the model parameters are stored too,
    and then the checkpoint is committed, which consists in updating the value
    of the checkpoint_version.txt file to contain the new version number. This
    scheme is chosen to work with shared filesystems (specifically, Gluster)
    which guarantee close/open data consistency but no metadata consistency (so
    os.rename is out).
    """

    def __init__(self, path: str) -> None:
        if path.startswith("file://"):
            path = path[len("file://"):]
        self.path: Path = Path(path).resolve(strict=False)

    def get_version_file(self, *, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "checkpoint_version.txt"

    def get_config_file(self, *, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "config.json"

    def get_entity_partition_file(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
        *,
        path: Optional[Path] = None,
    ) -> Path:
        if path is None:
            path = self.path
        return path / f"embeddings_{entity_name}_{partition}.v{version}.h5"

    def get_model_file(self, version: int, *, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / f"model.v{version}.h5"

    def get_snapshot_path(self, epoch_idx: int) -> Path:
        return self.path / f"epoch_{epoch_idx}"

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def save_version(self, version: int) -> None:
        with self.get_version_file().open("wt") as tf:
            tf.write(f"{version}\n")
            tf.flush()
            os.fsync(tf.fileno())

    def load_version(self) -> int:
        # FIXME: there's a slight danger here, say that a multi-machine job fails
        # after a few versions, and then it reruns but one of the write_version=False
        # machines has cached the metadata and thinks it doesn't exist, then it
        # will expect checkpoint_version=0 and fail.
        try:
            with self.get_version_file().open("rt") as tf:
                version_string = tf.read().strip()
        except FileNotFoundError:
            return 0
        else:
            # On some distributed filesystems creating the file (with an empty
            # content) and writing "0" to it are separate actions thus a race
            # condition could occur where trainers see the file as empty.
            if len(version_string) == 0:
                return 0
            else:
                return int(version_string)

    def save_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
        embs: FloatTensorType,
        optim_state: Optional[bytes],
        metadata: Dict[str, Any],
    ) -> None:
        path = self.get_entity_partition_file(version, entity_name, partition)
        logger.debug(f"Saving to {path}")
        with h5py.File(path, "w") as hf:
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
            for k, v in metadata.items():
                hf.attrs[k] = v
            save_embeddings(hf, embs)
            save_optimizer_state_dict(hf, optim_state)
            hf.flush()
        logger.debug(f"Done saving to {path}")

    def load_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
    ) -> Tuple[FloatTensorType, Optional[bytes]]:
        path = self.get_entity_partition_file(version, entity_name, partition)
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
            if f"errno = {errno.ENOENT}" in str(err):
                raise CouldNotLoadData() from err
            raise err
        logger.debug(f"Done loading from {path}")
        return embs, optim_state

    def drop_entity_partition(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
    ) -> None:
        path = self.get_entity_partition_file(version, entity_name, partition)
        if path.exists():
            path.unlink()

    def save_model(
        self,
        version: int,
        state_dict: Dict[str, ModelParameter],
        optim_state: Optional[bytes],
        metadata: Dict[str, Any],
    ) -> None:
        path = self.get_model_file(version)
        logger.debug(f"Saving to {path}")
        with h5py.File(path, "w") as hf:
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
            for k, v in metadata.items():
                hf.attrs[k] = v
            save_model_state_dict(hf, state_dict)
            save_optimizer_state_dict(hf, optim_state)
            hf.flush()
        logger.debug(f"Done saving to {path}")

    def load_model(
        self,
        version: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[bytes]]:
        path = self.get_model_file(version)
        logger.debug(f"Loading from {path}")
        try:
            with h5py.File(path, "r") as hf:
                if hf.attrs.get(FORMAT_VERSION_ATTR, None) != FORMAT_VERSION:
                    raise RuntimeError(f"Version mismatch in model file {path}")
                state_dict = load_model_state_dict(hf)
                optim_state = load_optimizer_state_dict(hf)
        except OSError as err:
            # h5py refuses to make it easy to figure out what went wrong. The errno
            # attribute is set to None. See https://github.com/h5py/h5py/issues/493.
            if f"errno = {errno.ENOENT}" in str(err):
                raise CouldNotLoadData() from err
            raise err
        logger.debug(f"Done loading from {path}")
        return state_dict, optim_state

    def drop_model(self, version: int) -> None:
        path = self.get_model_file(version)
        if path.exists():
            path.unlink()

    def save_config(self, config_json: str) -> None:
        with self.get_config_file().open("wt") as tf:
            tf.write(config_json)

    def load_config(self) -> str:
        with self.get_config_file().open("rt") as tf:
            return tf.read()

    def prepare_snapshot(self, version: int, epoch_idx: int) -> None:
        self.get_snapshot_path(epoch_idx).mkdir(parents=True, exist_ok=True)

    def copy_entity_partition_to_snapshot(
        self,
        version: int,
        entity_name: EntityName,
        partition: Partition,
        epoch_idx: int,
    ) -> None:
        src_path = self.get_entity_partition_file(version, entity_name, partition)
        dst_path = self.get_entity_partition_file(
            version, entity_name, partition, path=self.get_snapshot_path(epoch_idx))
        dst_path.symlink_to(src_path)

    def copy_model_to_snapshot(self, version: int, epoch_idx: int) -> None:
        src_path = self.get_model_file(version)
        dst_path = self.get_model_file(
            version, path=self.get_snapshot_path(epoch_idx))
        dst_path.symlink_to(src_path)

    def copy_version_to_snapshot(self, version: int, epoch_idx: int) -> None:
        dst_path = self.get_version_file(path=self.get_snapshot_path(epoch_idx))
        with dst_path.open("wt") as tf:
            tf.write(f"{version}\n")
