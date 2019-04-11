#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import io
import json
import os
import os.path
import shutil
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from glob import glob
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch_extensions.tensorlist.tensorlist import TensorList
from torch_extensions.rpc.rpc import (
    _serialize as torch_rpc_serialize,
    _deserialize as torch_rpc_deserialize,
)

from .config import ConfigSchema
from .entitylist import EntityList
from .parameter_sharing import ParameterClient
from .types import EntityName, Partition, Rank, OptimizerStateDict, ModuleStateDict, \
    FloatTensorType, LongTensorType
from .util import log, vlog, create_pool


def maybe_old_entity_path(path: str) -> bool:
    # We used to store them as pickles.
    return (bool(glob(os.path.join(path, "entity_count_*.pt")))
            or bool(glob(os.path.join(path, "dynamic_rel_count.pt"))))


def maybe_old_edge_path(path: str) -> bool:
    # We used to have 1-based indexing.
    return (os.path.exists(os.path.join(path, "edges_1_1.h5"))
            and not os.path.exists(os.path.join(path, "edges_0_0.h5")))


def maybe_old_checkpoint_path(path: str) -> bool:
    # We used to store them as pickles.
    return (bool(glob(os.path.join(path, "*CHECKPOINT_VERSION*")))
            or bool(glob(os.path.join(path, "*.pt*"))))


class EdgeReader:
    """Reads partitioned edgelists from disk, in the format
    created by edge_downloader.py.

    Edge lists are stored as hdf5 allowing partial reads (for multi-pass).

    Currently simple implementation but should eventually be multi-threaded /
    pipelined.
    """

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise RuntimeError("Invalid edge dir: %s" % path)
        if maybe_old_edge_path(path):
            log("WARNING: It may be that one of your edge paths contains files "
                "using the old format. See D14241362 for how to update them.")
        self.path: str = path

    def read(
        self,
        lhs_p: Partition,
        rhs_p: Partition,
        chunk_idx: int = 0,
        num_chunks: int = 1,
    ) -> Tuple[EntityList, EntityList, LongTensorType]:
        file_path = os.path.join(self.path, "edges_%d_%d.h5" % (lhs_p, rhs_p))
        assert os.path.exists(file_path), "%s does not exist" % file_path
        with h5py.File(file_path, 'r') as hf:
            if FORMAT_VERSION_ATTR not in hf.attrs:
                log("WARNING: It may be that one of your edge paths contains "
                    "files using the old format. See D14241362 for how to "
                    "update them.")
            elif hf.attrs[FORMAT_VERSION_ATTR] != FORMAT_VERSION:
                raise RuntimeError("Version mismatch in edge file %s" % file_path)
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

            return (EntityList(lhs, lhsd),
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
            offsets_ds = hf['%s_offsets' % key]
            data_ds = hf['%s_data' % key]
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

        offsets -= offsets[0]

        return TensorList(offsets, data)


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


class Mapping:

    def __init__(self, private: str, public: str, fields: List[str]) -> None:
        self.private_to_public = OneWayMapping(private.replace(".", r"\."), public, fields)
        self.public_to_private = OneWayMapping(public, private, fields)


MODEL_STATE_DICT_MAPPINGS = [
    Mapping(private="{side}_operators.{idx}.{param}",
            public="relations/{idx}/operator/{side}/{param}",
            fields=["idx", "side", "param"]),
    Mapping(private="global_embs.emb_{type}",
            public="entities/{type}/global_embedding",
            fields=["type"]),
]


def save_model_state_dict(hf: h5py.File, state_dict: ModuleStateDict) -> None:
    g = hf.create_group(MODEL_STATE_DICT_GROUP, track_order=True)
    for private_key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Isn't the state dict supposed to be "
                               "a shallow key-to-tensor mapping?!")
        for mapping in MODEL_STATE_DICT_MAPPINGS:
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


def load_model_state_dict(hf: h5py.File) -> Optional[ModuleStateDict]:
    if MODEL_STATE_DICT_GROUP not in hf:
        return None
    g = hf[MODEL_STATE_DICT_GROUP]
    state_dict = ModuleStateDict({})

    def process_dataset(public_key, dataset) -> None:
        if not isinstance(dataset, h5py.Dataset):
            return
        for mapping in MODEL_STATE_DICT_MAPPINGS:
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
    vlog("Saving to %s" % path)
    with h5py.File(path, "w") as hf:
        hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
        for k, v in metadata.items():
            hf.attrs[k] = v
        save_embeddings(hf, embs)
        save_optimizer_state_dict(hf, optim_state)
        hf.flush()
    vlog("Done saving to %s" % path)


def save_model(
    path: str,
    state_dict: ModuleStateDict,
    optim_state: Optional[OptimizerStateDict],
    metadata: Dict[str, Any],
) -> None:
    vlog("Saving to %s" % path)
    with h5py.File(path, "w") as hf:
        hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
        for k, v in metadata.items():
            hf.attrs[k] = v
        save_model_state_dict(hf, state_dict)
        save_optimizer_state_dict(hf, optim_state)
        hf.flush()
    vlog("Done saving to %s" % path)


def load_entity_partition(
    path: str,
) -> Tuple[FloatTensorType, Optional[OptimizerStateDict]]:
    vlog("Loading from %s" % path)
    try:
        with h5py.File(path, "r") as hf:
            if hf.attrs[FORMAT_VERSION_ATTR] != FORMAT_VERSION:
                raise RuntimeError("Version mismatch in embeddings file %s")
            embs = load_embeddings(hf)
            optim_state = load_optimizer_state_dict(hf)
    except OSError as err:
        # h5py refuses to make it easy to figure out what went wrong. The errno
        # attribute is set to None. See https://github.com/h5py/h5py/issues/493.
        if "errno = %d" % errno.ENOENT in str(err):
            raise FileNotFoundError() from err
        raise err
    vlog("Done loading from %s" % path)
    return embs, optim_state


def load_model(
    path: str,
) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
    vlog("Loading from %s" % path)
    try:
        with h5py.File(path, "r") as hf:
            if hf.attrs[FORMAT_VERSION_ATTR] != FORMAT_VERSION:
                raise RuntimeError("Version mismatch in model file %s")
            state_dict = load_model_state_dict(hf)
            optim_state = load_optimizer_state_dict(hf)
    except OSError as err:
        # h5py refuses to make it easy to figure out what went wrong. The errno
        # attribute is set to None. See https://github.com/h5py/h5py/issues/493.
        if "errno = %d" % errno.ENOENT in str(err):
            raise FileNotFoundError() from err
        raise err
    vlog("Done loading from %s" % path)
    return state_dict, optim_state


def noop() -> None:
    pass


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
        optim_state: Optional[OptimizerStateDict],
    ) -> None:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        client.store(key + "__embs", embs)
        client.store(key + "__optim", torch_rpc_serialize(optim_state))

    def get(
        self,
        entity: EntityName,
        part: Partition,
    ) -> Tuple[FloatTensorType, OptimizerStateDict]:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        embs = client.get(key + "__embs", shared=True)
        assert embs is not None
        optim_state = torch_rpc_deserialize(client.get(key + "__optim"))
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
    ) -> None:
        """
        Args:
          - path : path to the folder containing checkpoints.
          - background: if True, will do prefetch and store in a background
                        process
        """

        if maybe_old_checkpoint_path(path):
            log("WARNING: It may be that your checkpoint path (or your init "
                "path) contains files using the old format. See D14241362 for "
                "how to update them.")

        self.path: str = path
        self.dirty: Set[Tuple[EntityName, Partition]] = set()
        self.rank: Rank = rank
        self.num_machines: int = num_machines
        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(os.path.join(path, VERSION_FILE)):
                self._write_version_file(0)

        # FIXME: there's a slight danger here, say that a multi-machine job fails
        # after a few versions, and then it reruns but one of the write_version=False
        # machines has cached the metadata and thinks it doesn't exist, then it
        # will expect checkpoint_version=0 and fail.
        try:
            with open(os.path.join(self.path, VERSION_FILE), "rt") as tf:
                self.checkpoint_version = int(tf.read().strip())
        except FileNotFoundError:
            self.checkpoint_version = 0

        self.background: bool = background
        if self.background:
            self.pool: mp.Pool = create_pool(1)
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
        marker_key = "marker %d" % marker
        future_res = self.pool.apply_async(noop, ())
        self.outstanding[marker_key] = future_res

    def wait_for_marker(self, marker: int) -> None:
        marker_key = "marker %d" % marker
        if marker_key not in self.outstanding:
            return
        self._sync(marker_key)

    def _sync(self, sync_path: Optional[str] = None) -> None:
        assert self.background
        vlog("CheckpointManager=>_sync( %s )" % sync_path)
        vlog("outstanding= %s" % set(self.outstanding))
        while len(self.outstanding) > 0:
            path, future_res = self.outstanding.popitem(last=False)
            res = future_res.get()

            if res is not None:
                log("Setting prefetched %s; %d outstanding" %
                    (path, len(self.outstanding)))
                self.prefetched[path] = res

            if sync_path is not None and path == sync_path:
                break

    def _version(self, dirty: bool = False) -> int:
        version = self.checkpoint_version
        if dirty:
            version += 1
        return version

    def _file_path(self, entity: EntityName, part: Partition) -> str:
        version = self._version((entity, part) in self.dirty)
        file_path = os.path.join(self.path, "embeddings_%s_%d.v%d.h5" % (entity, part, version))
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

        file_path = self._file_path(entity, part)

        if self.background:
            self._sync(file_path)

        metadata = self.collect_metadata()

        if self.partition_client is not None:
            self.partition_client.store(entity, part, embs, optim_state)
        elif self.background:
            if file_path in self.prefetched:
                self.prefetched.pop(file_path)
            future_res = self.pool.apply_async(
                save_entity_partition, (file_path, embs, optim_state, metadata))
            self.outstanding[file_path] = future_res
        else:
            save_entity_partition(file_path, embs, optim_state, metadata)

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

        file_path = self._file_path(entity, part)
        if (entity, part) in self.dirty and self.partition_client is not None:
            return self.partition_client.get(entity, part)
        if self.background:
            self._sync(file_path)
            if file_path in self.prefetched:
                return self.prefetched.pop(file_path)
        return load_entity_partition(file_path)

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

        file_path = self._file_path(entity, part)
        if file_path in self.outstanding or file_path in self.prefetched:
            vlog("Bailing from prefetch of %s" % file_path)
            return
        if os.path.exists(file_path):
            future_res = self.pool.apply_async(load_entity_partition, (file_path,))
            self.outstanding[file_path] = future_res

    def write_model(
        self,
        model_state: ModuleStateDict,
        optim_state: Optional[OptimizerStateDict],
    ) -> None:
        version = self._version(True)
        file_path = os.path.join(self.path, "model.v%d.h5" % version)
        metadata = self.collect_metadata()
        save_model(file_path, model_state, optim_state, metadata)

    def read_model(
        self,
    ) -> Tuple[Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        version = self._version(False)
        file_path = os.path.join(self.path, "model.v%d.h5" % version)
        return load_model(file_path)

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
                    vlog("Rank %d: getting %s %d" % (self.rank, entity, part))
                    embs, optim_state = \
                        self.partition_client.get(EntityName(entity), Partition(part))
                    vlog("Rank %d: saving %s %d to disk" % (self.rank, entity, part))
                    new_file_path = os.path.join(
                        self.path, "embeddings_%s_%d.v%d.h5" % (entity, part, new_version))
                    save_entity_partition(new_file_path, embs, optim_state, metadata)

    def switch_to_new_version(self) -> None:
        self.dirty.clear()
        self.checkpoint_version += 1
        if self.rank == 0:
            vlog("Rank 0: write version file")
            self._write_version_file(self.checkpoint_version)
            vlog("Rank 0: done")

    def remove_old_version(self, config: ConfigSchema) -> None:
        old_version = self.checkpoint_version - 1
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                old_file_path = os.path.join(
                    self.path, "embeddings_%s_%d.v%d.h5" % (entity, part, old_version))
                vlog("%d os.remove %s" % (self.rank, old_file_path))
                if self.checkpoint_version > 1 or os.path.exists(old_file_path):
                    os.remove(old_file_path)

    def save_current_version(
        self,
        config: ConfigSchema,
        epoch_idx: int
    ) -> None:
        """
        This function merely moves all files in current version
        into a separate folder a symlink is left behind

        ? random note: epoch_idx can be replaced with anything, here we just
        use this as a postfix
        """
        save_dir = os.path.join(self.path, 'epoch_%d' % epoch_idx)
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                file_name = "embeddings_%s_%d.v%d.h5" % (entity, part, self.checkpoint_version)
                src_path = os.path.join(self.path, file_name)
                dst_path = os.path.join(save_dir, file_name)
                if os.path.exists(src_path):
                    os.makedirs(save_dir, exist_ok=True)
                    shutil.move(src_path, dst_path)

    def close(self) -> None:
        if self.background:
            self.pool.close()
            self.pool.join()

    def join(self) -> None:
        # FIXME: this whole join thing doesn't work with torch.distributed
        # can just get rid of it
        if self.partition_client is not None:
            self.partition_client.join()
