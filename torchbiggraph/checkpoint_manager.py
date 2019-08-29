#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import io
import json
import logging
import multiprocessing as mp
import multiprocessing.pool  # noqa: F401
import os
import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.multiprocessing

from torchbiggraph.checkpoint_storage import (
    load_entity_partition,
    load_model,
    TwoWayMapping,
    save_entity_partition,
    save_model,
)
from torchbiggraph.config import ConfigSchema
from torchbiggraph.parameter_sharing import ParameterClient
from torchbiggraph.types import (
    ByteTensorType,
    EntityName,
    FloatTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
)
from torchbiggraph.util import create_pool, get_async_result


logger = logging.getLogger("torchbiggraph")


MODEL_STATE_DICT_MAPPINGS = [
    TwoWayMapping(private="{side}_operators.{idx}.{param}",
                  public="relations/{idx}/operator/{side}/{param}",
                  fields=["idx", "side", "param"]),
    TwoWayMapping(private="global_embs.emb_{type}",
                  public="entities/{type}/global_embedding",
                  fields=["type"]),
]


def noop() -> None:
    pass


def bytes_to_bytetensor(data: bytes) -> ByteTensorType:
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8))


def bytetensor_to_bytes(tensor: ByteTensorType) -> bytes:
    return tensor.numpy().tobytes()


VERSION_FILE = "checkpoint_version.txt"
CONFIG_FILE = "config.json"


class PartitionClient:
    """A wrapper around ParameterClient that knows how to read and write
    partitions (i.e. pairs (embs, optim_state)) to the parameter servers.
    """

    def __init__(self, server_ranks: List[Rank]) -> None:
        self._clients = [ParameterClient(rank) for rank in server_ranks]

    def store(
        self,
        entity: EntityName,
        part: Partition,
        embs: FloatTensorType,
        optim_state: Optional[bytes],
    ) -> None:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        client.store(key + "__embs", embs)
        if optim_state is not None:
            optim_state_tensor = bytes_to_bytetensor(optim_state)
            client.store(key + "__optim", optim_state_tensor)

    def get(
        self,
        entity: EntityName,
        part: Partition,
    ) -> Tuple[FloatTensorType, Optional[bytes]]:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        embs = client.get(key + "__embs", shared=True)
        assert embs is not None
        optim_state_tensor = client.get(key + "__optim")
        if optim_state_tensor is not None:
            optim_state = bytetensor_to_bytes(optim_state_tensor)
        else:
            optim_state = None
        return embs, optim_state

    def join(self) -> None:
        for client in self._clients:
            client.join()


class MetadataProvider(ABC):

    @abstractmethod
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        pass


class ConfigMetadataProvider(MetadataProvider):

    def __init__(self, config: ConfigSchema) -> None:
        self.json_config_dict = json.dumps(config.to_dict(), indent=4)

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {"config/json": self.json_config_dict}


def serialize_optim_state(
    optim_state: Optional[OptimizerStateDict],
) -> Optional[bytes]:
    if optim_state is None:
        return None
    with io.BytesIO() as bf:
        torch.save(optim_state, bf)
        return bf.getvalue()


def deserialize_optim_state(
    serialized_optim_state: Optional[bytes],
) -> Optional[OptimizerStateDict]:
    if serialized_optim_state is None:
        return None
    with io.BytesIO(serialized_optim_state) as bf:
        return torch.load(bf)


class CheckpointManager:
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

    def __init__(
        self,
        path: str,
        rank: Rank = -1,
        num_machines: int = 1,
        background: bool = False,
        partition_client: Optional[PartitionClient] = None,
        subprocess_name: Optional[str] = None,
        subprocess_init: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Args:
          - path : path to the folder containing checkpoints.
          - background: if True, will do prefetch and store in a background
                        process
        """
        self.path: str = os.path.abspath(path)
        self.dirty: Set[Tuple[EntityName, Partition]] = set()
        self.rank: Rank = rank
        self.num_machines: int = num_machines
        if self.rank == 0:
            os.makedirs(self.path, exist_ok=True)

        # FIXME: there's a slight danger here, say that a multi-machine job fails
        # after a few versions, and then it reruns but one of the write_version=False
        # machines has cached the metadata and thinks it doesn't exist, then it
        # will expect checkpoint_version=0 and fail.
        try:
            with open(os.path.join(self.path, VERSION_FILE), "rt") as tf:
                version_string = tf.read().strip()
        except FileNotFoundError:
            self.checkpoint_version = 0
        else:
            # On some distributed filesystems creating the file (with an empty
            # content) and writing "0" to it are separate actions thus a race
            # condition could occur where trainers see the file as empty.
            if len(version_string) == 0:
                self.checkpoint_version = 0
            else:
                self.checkpoint_version = int(version_string)

        self.background: bool = background
        if self.background:
            self.pool: mp.pool.Pool = create_pool(
                1,
                subprocess_name=subprocess_name,
                subprocess_init=subprocess_init,
            )
            # FIXME In py-3.7 switch to typing.OrderedDict[str, AsyncResult].
            self.outstanding: OrderedDict = OrderedDict()
            self.prefetched: Dict[str, Tuple[FloatTensorType, Optional[OptimizerStateDict]]] = {}

        self.partition_client = partition_client

        self.metadata_providers: List[MetadataProvider] = []

    def register_metadata_provider(self, provider: MetadataProvider) -> None:
        self.metadata_providers.append(provider)

    def collect_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for provider in self.metadata_providers:
            metadata.update(provider.get_checkpoint_metadata())
        return metadata

    def record_marker(self, marker: int) -> None:
        assert self.background
        marker_token = f"marker {marker}"
        future_res = self.pool.apply_async(noop, ())
        self.outstanding[marker_token] = future_res

    def wait_for_marker(self, marker: int) -> None:
        marker_token = f"marker {marker}"
        if marker_token not in self.outstanding:
            return
        self._sync(marker_token)

    def _sync(self, sync_token: Optional[str] = None) -> None:
        assert self.background
        logger.debug(f"CheckpointManager=>_sync( {sync_token} )")
        logger.debug(f"outstanding= {set(self.outstanding)}")
        while len(self.outstanding) > 0:
            token, future_res = self.outstanding.popitem(last=False)
            res = get_async_result(future_res, self.pool)

            if res is not None:
                logger.info(
                    f"Setting prefetched {token}; {len(self.outstanding)} outstanding")
                self.prefetched[token] = res

            if sync_token is not None and token == sync_token:
                break

    def _version(self, dirty: bool = False) -> int:
        version = self.checkpoint_version
        if dirty:
            version += 1
        return version

    def _file_path(self, entity: EntityName, part: Partition) -> str:
        version = self._version((entity, part) in self.dirty)
        file_path = os.path.join(self.path, f"embeddings_{entity}_{part}.v{version}.h5")
        return file_path

    def write(
        self,
        entity: EntityName,
        part: Partition,
        embs: FloatTensorType,
        optim_state: Optional[OptimizerStateDict],
        force_clean: bool = False,
    ) -> None:
        if not force_clean:
            self.dirty.add((entity, part))

        version = self._version((entity, part) in self.dirty)
        token = f"entity {entity} {part} v{version}"
        file_path = self._file_path(entity, part)

        if self.background:
            self._sync(token)

        metadata = self.collect_metadata()
        serialized_optim_state = serialize_optim_state(optim_state)

        if self.partition_client is not None:
            self.partition_client.store(entity, part, embs, serialized_optim_state)
        elif self.background:
            if token in self.prefetched:
                self.prefetched.pop(token)
            future_res = self.pool.apply_async(
                save_entity_partition, (file_path, embs, serialized_optim_state, metadata))
            self.outstanding[token] = future_res
        else:
            save_entity_partition(file_path, embs, serialized_optim_state, metadata)

    def read(
        self,
        entity: EntityName,
        part: Partition,
        *,
        force_dirty: bool = False,
    ) -> Tuple[FloatTensorType, Optional[OptimizerStateDict]]:
        # if counter > 1, we are inside a pass. Otherwise, we are doing
        # evals or just finished an epoch so ".new" files do not exist.
        if force_dirty:
            self.dirty.add((entity, part))

        version = self._version((entity, part) in self.dirty)
        token = f"entity {entity} {part} v{version}"
        file_path = self._file_path(entity, part)
        if (entity, part) in self.dirty and self.partition_client is not None:
            embs, serialized_optim_state = self.partition_client.get(entity, part)
        elif self.background:
            self._sync(token)
            if token in self.prefetched:
                embs, serialized_optim_state = self.prefetched.pop(token)
            else:
                embs, serialized_optim_state = load_entity_partition(file_path)
        else:
            embs, serialized_optim_state = load_entity_partition(file_path)
        optim_state = deserialize_optim_state(serialized_optim_state)
        return embs, optim_state

    def maybe_read(
        self,
        entity: EntityName,
        part: Partition,
        *,
        force_dirty: bool = False,
    ) -> Tuple[Optional[FloatTensorType], Optional[OptimizerStateDict]]:
        try:
            return self.read(entity, part, force_dirty=force_dirty)
        except FileNotFoundError:
            # if it's dirty then we've already written this file, so it should exist
            if (entity, part) in self.dirty:
                raise
            return None, None

    def prefetch(
        self,
        entity: EntityName,
        part: Partition,
    ) -> None:
        if not self.background:
            return

        version = self._version((entity, part) in self.dirty)
        token = f"entity {entity} {part} v{version}"
        file_path = self._file_path(entity, part)
        if token in self.outstanding or token in self.prefetched:
            logger.debug(f"Bailing from prefetch of {token}")
            return
        if os.path.exists(file_path):
            future_res = self.pool.apply_async(load_entity_partition, (file_path,))
            self.outstanding[token] = future_res

    def write_model(
        self,
        model_state: ModuleStateDict,
        optim_state: Optional[OptimizerStateDict],
    ) -> None:
        version = self._version(True)
        file_path = os.path.join(self.path, f"model.v{version}.h5")
        metadata = self.collect_metadata()
        serialized_optim_state = serialize_optim_state(optim_state)
        save_model(file_path, model_state, serialized_optim_state, metadata, MODEL_STATE_DICT_MAPPINGS)

    def read_model(
        self,
    ) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        version = self._version(False)
        file_path = os.path.join(self.path, f"model.v{version}.h5")
        state_dict, serialized_optim_state = load_model(file_path, MODEL_STATE_DICT_MAPPINGS)
        optim_state = deserialize_optim_state(serialized_optim_state)
        return state_dict, optim_state

    def maybe_read_model(
        self,
    ) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        try:
            return self.read_model()
        except FileNotFoundError:
            return None, None

    def write_config(self, config: ConfigSchema) -> None:
        with open(os.path.join(self.path, CONFIG_FILE), "wt") as tf:
            json.dump(config.to_dict(), tf, indent=4)

    def read_config(self) -> ConfigSchema:
        with open(os.path.join(self.path, CONFIG_FILE), "rt") as tf:
            return ConfigSchema.from_dict(json.load(tf))

    def _write_version_file(self, version: int) -> None:
        with open(os.path.join(self.path, VERSION_FILE), "wt") as tf:
            tf.write("%d" % version)
            tf.flush()
            os.fsync(tf.fileno())

    def write_new_version(self, config: ConfigSchema) -> None:
        if self.background:
            self._sync()
        metadata = self.collect_metadata()
        new_version = self._version(True)
        if self.partition_client is not None:
            for entity, econf in config.entities.items():
                for part in range(self.rank, econf.num_partitions, self.num_machines):
                    logger.debug(f"Getting {entity} {part}")
                    embs, serialized_optim_state = \
                        self.partition_client.get(EntityName(entity), Partition(part))
                    logger.debug(f"Done getting {entity} {part}")
                    new_file_path = os.path.join(
                        self.path, f"embeddings_{entity}_{part}.v{new_version}.h5")
                    logger.debug(f"Saving {entity} {part} to {new_file_path}")
                    save_entity_partition(new_file_path, embs, serialized_optim_state, metadata)
                    logger.debug(f"Done saving {entity} {part} to {new_file_path}")

    def switch_to_new_version(self) -> None:
        self.dirty.clear()
        self.checkpoint_version += 1
        if self.rank == 0:
            logger.debug("Writing version file")
            self._write_version_file(self.checkpoint_version)
            logger.debug("Written version file")

    def remove_old_version(self, config: ConfigSchema) -> None:
        old_version = self.checkpoint_version - 1
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                old_file_path = os.path.join(
                    self.path, f"embeddings_{entity}_{part}.v{old_version}.h5")
                if self.checkpoint_version > 1 or os.path.exists(old_file_path):
                    os.remove(old_file_path)
                    logger.debug(f"Done deleting {old_file_path}")
        if self.rank == 0:
            old_file_path = os.path.join(self.path, f"model.v{old_version}.h5")
            if self.checkpoint_version > 1 or os.path.exists(old_file_path):
                logger.debug(f"Deleting {old_file_path}")
                os.remove(old_file_path)
                logger.debug(f"Done deleting {old_file_path}")

    def preserve_current_version(self, config: ConfigSchema, epoch_idx: int) -> None:
        """Create a snapshot (made of symlinks) of the current checkpoint

        In order to "archive" a checkpoint version this function creates a
        subdirectory (named after the given epoch) and recreates the current
        checkpoint using symlinks to the originals (they cannot be moved as they
        may still be needed, and copying would take long and waste space).
        """
        version = self.checkpoint_version
        dst_dir = os.path.join(self.path, f"epoch_{epoch_idx}")
        os.makedirs(dst_dir, exist_ok=True)
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                file_name = f"embeddings_{entity}_{part}.v{version}.h5"
                src_file_path = os.path.join(self.path, file_name)
                dst_file_path = os.path.join(dst_dir, file_name)
                os.symlink(src_file_path, dst_file_path)
        if self.rank == 0:
            file_name = f"model.v{version}.h5"
            src_file_path = os.path.join(self.path, file_name)
            dst_file_path = os.path.join(dst_dir, file_name)
            os.symlink(src_file_path, dst_file_path)
            with open(os.path.join(dst_dir, VERSION_FILE), "xt") as tf:
                tf.write(f"{version}")

    def close(self) -> None:
        if self.background:
            self.pool.close()
            self.pool.join()

    def join(self) -> None:
        # FIXME: this whole join thing doesn't work with torch.distributed
        # can just get rid of it
        if self.partition_client is not None:
            self.partition_client.join()
