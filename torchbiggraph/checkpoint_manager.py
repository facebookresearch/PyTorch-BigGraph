#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import io
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.distributed as td
import torch.multiprocessing
from torchbiggraph.checkpoint_storage import (
    CHECKPOINT_STORAGES,
    AbstractCheckpointStorage,
    ModelParameter,
)
from torchbiggraph.config import ConfigSchema
from torchbiggraph.parameter_sharing import ParameterClient
from torchbiggraph.stats import SerializedStats
from torchbiggraph.types import (
    SINGLE_TRAINER,
    ByteTensorType,
    EntityName,
    FloatTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
)
from torchbiggraph.util import CouldNotLoadData


logger = logging.getLogger("torchbiggraph")


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
        self.private_to_public = OneWayMapping(
            private.replace(".", r"\."), public, fields
        )
        self.public_to_private = OneWayMapping(public, private, fields)


MODEL_STATE_DICT_MAPPINGS = [
    TwoWayMapping(
        private="{side}_operators.{idx}.{param}",
        public="relations/{idx}/operator/{side}/{param}",
        fields=["idx", "side", "param"],
    ),
    TwoWayMapping(
        private="global_embs.emb_{type}",
        public="entities/{type}/global_embedding",
        fields=["type"],
    ),
]


def model_state_dict_public_to_private(
    public_state_dict: Optional[Dict[str, torch.Tensor]]
) -> Optional[ModuleStateDict]:
    if public_state_dict is None:
        return None
    private_state_dict = {}
    for public_name, tensor in public_state_dict.items():
        for mapping in MODEL_STATE_DICT_MAPPINGS:
            try:
                private_name = mapping.public_to_private.map(public_name)
            except ValueError:
                continue
            else:
                break
        else:
            raise RuntimeError(f"Couldn't find a match for dataset name: {public_name}")
        private_state_dict[private_name] = tensor
    return private_state_dict


def model_state_dict_private_to_public(
    private_state_dict: ModuleStateDict,
) -> Dict[str, ModelParameter]:
    public_state_dict: Dict[str, ModelParameter] = {}
    for private_name, tensor in private_state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                "Isn't the state dict supposed to be "
                "a shallow key-to-tensor mapping?!"
            )
        for mapping in MODEL_STATE_DICT_MAPPINGS:
            try:
                public_name = mapping.private_to_public.map(private_name)
            except ValueError:
                continue
            else:
                break
        else:
            raise RuntimeError(
                f"Couldn't find a match for state dict key: {private_name}"
            )
        public_state_dict[public_name] = ModelParameter(private_name, tensor)
    return public_state_dict


def noop() -> None:
    pass


def bytes_to_bytetensor(data: bytes) -> ByteTensorType:
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8))


def bytetensor_to_bytes(tensor: ByteTensorType) -> bytes:
    return tensor.numpy().tobytes()


class PartitionClient:
    """A wrapper around ParameterClient that knows how to read and write
    partitions (i.e. pairs (embs, optim_state)) to the parameter servers.
    """

    def __init__(
        self,
        server_ranks: List[Rank],
        groups: Optional[List["td.ProcessGroup"]] = None,
        log_stats: bool = False,
    ) -> None:
        self.groups = groups
        self._clients = [
            ParameterClient(rank, groups, log_stats) for rank in server_ranks
        ]

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
        self, entity: EntityName, part: Partition, out: Optional[FloatTensorType] = None
    ) -> Tuple[FloatTensorType, Optional[bytes]]:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        embs = client.get(key + "__embs", dst=out, shared=True)
        assert embs is not None
        optim_state_tensor = client.get(key + "__optim")
        if optim_state_tensor is not None:
            optim_state = bytetensor_to_bytes(optim_state_tensor)
        else:
            optim_state = None
        return embs, optim_state

    def join(self) -> None:
        for client in self._clients:
            client.join(do_barrier=True)
        logger.info("PartitionClient join barrier start")
        td.barrier(self.groups[0])
        logger.info("PartitionClient join barrier end")


class MetadataProvider(ABC):
    @abstractmethod
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        pass


class ConfigMetadataProvider(MetadataProvider):
    def __init__(self, config: ConfigSchema) -> None:
        self.json_config_dict = json.dumps(config.to_dict(), indent=4)

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {"config/json": self.json_config_dict}


def serialize_optim_state(optim_state: Optional[OptimizerStateDict]) -> Optional[bytes]:
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
    def __init__(
        self,
        url: str,
        rank: Rank = SINGLE_TRAINER,
        num_machines: int = 1,
        partition_client: Optional[PartitionClient] = None,
        subprocess_name: Optional[str] = None,
        subprocess_init: Optional[Callable[[], None]] = None,
    ) -> None:
        self.storage: AbstractCheckpointStorage = CHECKPOINT_STORAGES.make_instance(url)
        self.dirty: Set[Tuple[EntityName, Partition]] = set()
        self.rank: Rank = rank
        self.num_machines: int = num_machines
        if self.rank == 0:
            self.storage.prepare()

        self.checkpoint_version = self.storage.load_version()

        self.partition_client = partition_client

        self.metadata_providers: List[MetadataProvider] = []

    def register_metadata_provider(self, provider: MetadataProvider) -> None:
        self.metadata_providers.append(provider)

    def collect_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for provider in self.metadata_providers:
            metadata.update(provider.get_checkpoint_metadata())
        return metadata

    def _version(self, dirty: bool = False) -> int:
        version = self.checkpoint_version
        if dirty:
            version += 1
        return version

    def write(
        self,
        entity: EntityName,
        part: Partition,
        embs: FloatTensorType,
        optim_state: Optional[OptimizerStateDict],
        force_clean: bool = False,
        unpartitioned: bool = False,
    ) -> None:
        if not force_clean:
            self.dirty.add((entity, part))

        version = self._version((entity, part) in self.dirty)

        metadata = self.collect_metadata()
        serialized_optim_state = serialize_optim_state(optim_state)

        if self.partition_client is not None and not unpartitioned:
            self.partition_client.store(entity, part, embs, serialized_optim_state)
        else:
            self.storage.save_entity_partition(
                version, entity, part, embs, serialized_optim_state, metadata
            )

    def read(
        self,
        entity: EntityName,
        part: Partition,
        out: Optional[FloatTensorType] = None,
        *,
        force_dirty: bool = False,
    ) -> Tuple[FloatTensorType, Optional[OptimizerStateDict]]:
        # if counter > 1, we are inside a pass. Otherwise, we are doing
        # evals or just finished an epoch so ".new" files do not exist.
        if force_dirty:
            self.dirty.add((entity, part))

        version = self._version((entity, part) in self.dirty)
        if (entity, part) in self.dirty and self.partition_client is not None:
            embs, serialized_optim_state = self.partition_client.get(entity, part, out)
        else:
            embs, serialized_optim_state = self.storage.load_entity_partition(
                version, entity, part, out
            )
        optim_state = deserialize_optim_state(serialized_optim_state)
        return embs, optim_state

    def maybe_read(
        self,
        entity: EntityName,
        part: Partition,
        out: Optional[FloatTensorType] = None,
        *,
        force_dirty: bool = False,
    ) -> Tuple[Optional[FloatTensorType], Optional[OptimizerStateDict]]:
        try:
            return self.read(entity, part, out=out, force_dirty=force_dirty)
        except CouldNotLoadData:
            # if it's dirty then we've already written this file, so it should exist
            if (entity, part) in self.dirty:
                raise
            return None, None

    def write_model(
        self, state_dict: ModuleStateDict, optim_state: Optional[OptimizerStateDict]
    ) -> None:
        version = self._version(True)
        metadata = self.collect_metadata()
        public_state_dict = model_state_dict_private_to_public(state_dict)
        serialized_optim_state = serialize_optim_state(optim_state)
        self.storage.save_model(
            version, public_state_dict, serialized_optim_state, metadata
        )

    def read_model(
        self,
    ) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        version = self._version(False)
        public_state_dict, serialized_optim_state = self.storage.load_model(version)
        state_dict = model_state_dict_public_to_private(public_state_dict)
        optim_state = deserialize_optim_state(serialized_optim_state)
        return state_dict, optim_state

    def maybe_read_model(
        self,
    ) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        try:
            return self.read_model()
        except CouldNotLoadData:
            return None, None

    def write_config(self, config: ConfigSchema) -> None:
        config_json = json.dumps(config.to_dict(), indent=4)
        self.storage.save_config(config_json)

    def read_config(self) -> ConfigSchema:
        config_json = self.storage.load_config()
        return ConfigSchema.from_dict(json.loads(config_json))

    def append_stats(self, stats: List[Dict[str, Union[int, SerializedStats]]]) -> None:
        self.storage.append_stats([json.dumps(s) for s in stats])

    def read_stats(
        self,
    ) -> Generator[Dict[str, Union[int, SerializedStats]], None, None]:
        for line in self.storage.load_stats():
            yield json.loads(line)

    def maybe_read_stats(
        self,
    ) -> Generator[Dict[str, Union[int, SerializedStats]], None, None]:
        try:
            yield from self.read_stats()
        except CouldNotLoadData:
            pass

    def write_new_version(
        self,
        config: ConfigSchema,
        entity_counts: Dict[EntityName, List[int]],
        embedding_storage_freelist: Dict[EntityName, Set[torch.FloatStorage]],
    ) -> None:
        metadata = self.collect_metadata()
        new_version = self._version(True)
        if self.partition_client is not None:
            for entity, econf in config.entities.items():
                dimension = config.entity_dimension(entity)

                if econf.num_partitions == 1:
                    # unpartitioned entities are not stored on the partition
                    # server; they are checkpointed separately in train.py
                    continue

                for part in range(self.rank, econf.num_partitions, self.num_machines):
                    logger.debug(f"Getting {entity} {part}")
                    count = entity_counts[entity][part]
                    s = next(iter(embedding_storage_freelist[entity]))
                    out = torch.FloatTensor(s).view(-1, dimension)[:count]
                    embs, serialized_optim_state = self.partition_client.get(
                        entity, part, out=out
                    )
                    logger.debug(f"Done getting {entity} {part}")
                    logger.debug(f"Saving {entity} {part} v{new_version}")
                    self.storage.save_entity_partition(
                        new_version,
                        entity,
                        part,
                        embs,
                        serialized_optim_state,
                        metadata,
                    )
                    logger.debug(f"Done saving {entity} {part} v{new_version}")

    def switch_to_new_version(self) -> None:
        self.dirty.clear()
        self.checkpoint_version += 1
        if self.rank == 0:
            logger.debug(f"Setting version to {self.checkpoint_version}")
            self.storage.save_version(self.checkpoint_version)
            logger.debug(f"Done setting version to {self.checkpoint_version}")

    def remove_old_version(self, config: ConfigSchema) -> None:
        old_version = self.checkpoint_version - 1
        # We never create a v0 checkpoint, so if there is one we leave it there.
        if old_version == 0:
            return
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                logger.debug(f"Deleting {entity} {part} v{old_version}")
                self.storage.drop_entity_partition(old_version, entity, part)
                logger.debug(f"Done deleting {entity} {part} v{old_version}")
        if self.rank == 0:
            logger.debug(f"Deleting model v{old_version}")
            self.storage.drop_model(old_version)
            logger.debug(f"Done deleting model v{old_version}")

    def preserve_current_version(self, config: ConfigSchema, epoch_idx: int) -> None:
        """Create a snapshot (made of symlinks) of the current checkpoint

        In order to "archive" a checkpoint version this function creates a
        subdirectory (named after the given epoch) and recreates the current
        checkpoint using symlinks to the originals (they cannot be moved as they
        may still be needed, and copying would take long and waste space).
        """
        version = self.checkpoint_version
        self.storage.prepare_snapshot(version, epoch_idx)
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                self.storage.copy_entity_partition_to_snapshot(
                    version, entity, part, epoch_idx
                )
        if self.rank == 0:
            self.storage.copy_model_to_snapshot(version, epoch_idx)
            self.storage.copy_version_to_snapshot(version, epoch_idx)

    def close(self) -> None:
        self.join()

    def join(self) -> None:
        if self.partition_client is not None:
            self.partition_client.join()
