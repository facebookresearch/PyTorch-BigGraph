#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import h5py
import torch
import torch.multiprocessing as mp
from torch_extensions.tensorlist.tensorlist import TensorList
from torch_extensions.rpc.rpc import (
    _serialize as torch_rpc_serialize,
    _deserialize as torch_rpc_deserialize,
)

from .config import ConfigSchema
from .entitylist import EntityList
from .util import log, vlog, create_pool, EntityName, Partition, Rank, \
    OptimizerStateDict, ModuleStateDict


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
        self.path: str = path

    def read(
        self,
        lhsP: Partition,
        rhsP: Partition,
        i: int = 0,
        N: int = 1,
    ) -> Tuple[EntityList, EntityList, torch.LongTensor]:
        file_path = os.path.join(self.path, "edges_%d_%d.h5" % (lhsP, rhsP))
        assert os.path.exists(file_path), "%s does not exist" % file_path
        with h5py.File(file_path, 'r') as f:
            lhs = f['lhs']
            rhs = f['rhs']
            rel = f['rel']

            L = rel.len()
            begin = int(i * L / N)
            end = int((i + 1) * L / N)

            lhs = torch.from_numpy(lhs[begin:end])
            rhs = torch.from_numpy(rhs[begin:end])
            rel = torch.from_numpy(rel[begin:end])
            lhsd = self.read_dynamic(f, 'lhsd', begin, end)
            rhsd = self.read_dynamic(f, 'rhsd', begin, end)

            return (EntityList(lhs, lhsd),
                    EntityList(rhs, rhsd),
                    rel)

    def read_dynamic(
        self,
        f: h5py.File,
        key: str,
        begin: int,
        end: int,
    ) -> TensorList:
        offsets_field = '%s_offsets' % key
        data_field = '%s_data' % key
        if offsets_field in f and data_field in f:
            offsets = torch.from_numpy(f[offsets_field][begin:end + 1]).long()
            data = torch.from_numpy(f[data_field][offsets[0]:offsets[-1]]).long()

            # careful! as of Pytorch 0.4, offsets[0] is a 0d tensor view, so
            # offsets -= offsets[0] will give the wrong result
            offsets -= offsets[0].item()

            return TensorList(offsets, data)
        else:
            # Empty tensor_list representation
            return TensorList(
                torch.zeros(end - begin + 1).long(), torch.Tensor([])
            )


EntityPartitionType = Tuple[torch.FloatTensor, OptimizerStateDict]
MetadataType = Tuple[ConfigSchema, ModuleStateDict, OptimizerStateDict]


def _torch_save(
    data: Any,
    file_path: str,
) -> None:
    with open(file_path, 'wb') as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())


def torch_load_shared(filename: str) -> Any:
    """Perform torch.load, with copy-free load of FloatTensors into shared memory."""

    # `torch.load` calls the `map_location` function on `storage` immediately
    # after it is constructed (before it is written to). Since the storage is not
    # accessed, no physical memory is allocated to the original until after it is moved to
    # shared memory.
    #
    # I construct a new storage rather than calling `share_memory_()` because
    # it avoids an extra copy of the uninitialized tensor into shared memory,
    # and is thus almost 2x faster.
    #
    # Note: an alternative (if this stops working) is to temporarily override
    # torch.FloatStorage.__new__ to point to _new_shared, then call `torch.load`.

    return torch.load(filename,
                      map_location=lambda storage, _: storage._new_shared(storage.size()))


def save_entity_partition(
    path: str,
    embs: torch.FloatTensor,
    optim_state: OptimizerStateDict,
) -> None:
    vlog("Saving to %s" % path)
    _torch_save((embs, optim_state), path)
    vlog("Done saving to %s" % path)


def save_metadata(
    path: str,
    config: ConfigSchema,
    state_dict: ModuleStateDict,
    optim_state: OptimizerStateDict,
) -> None:
    vlog("Saving to %s" % path)
    _torch_save((config.to_dict(), state_dict, optim_state), path)
    vlog("Done saving to %s" % path)


def load_entity_partition(path: str) -> EntityPartitionType:
    vlog("Loading from %s" % path)
    embs, optim_state = torch_load_shared(path)
    vlog("Done loading from %s" % path)
    return embs, optim_state


def load_metadata(path: str) -> MetadataType:
    vlog("Loading from %s" % path)
    config_dict, state_dict, optim_state = torch_load_shared(path)
    vlog("Done loading from %s" % path)
    config = ConfigSchema.from_dict(config_dict)
    return config, state_dict, optim_state


def noop() -> None:
    pass


VERSION_FILE = "checkpoint_version.txt"


class PartitionClient:
    """A wrapper around ParameterServerClient that knows how to read and write
    partitions (i.e. pairs (embs, optim_state)) to the parameter servers.
    """

    def __init__(self, server_ranks: Iterable[int]) -> None:
        from .parameterserver import ParameterServerClient
        self._clients = [ParameterServerClient(rank) for rank in server_ranks]

    def store(
        self,
        entity: EntityName,
        part: Partition,
        embs: torch.FloatTensor,
        optim_state: OptimizerStateDict,
    ) -> None:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        client.store(key + "__embs", embs)
        client.store(key + "__optim", torch_rpc_serialize(optim_state))

    def get(
        self,
        entity: EntityName,
        part: Partition,
    ) -> EntityPartitionType:
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        embs = client.get(key + "__embs", shared=True)
        assert embs is not None
        optim_state = torch_rpc_deserialize(client.get(key + "__optim"))
        return embs, optim_state

    def join(self) -> None:
        for client in self._clients:
            client.join()


class CheckpointManager:
    """Reads and writes checkpoint data to/from disk.

    Checkpoints are saved via torch.save with no introspection into the
    contents of the data.

    Swapped-out partitions are saved to disk with a temporary extension, and when
    `commit()` is called, that extension is marked as the 'checkpoint' extension
    via the checkpoint_version.txt file. This scheme is chosen to work with shared
    filesystems (specifically, Gluster) which guarantee close/open data consistency
    but no metadata consistency (so os.rename is out).

    """

    def __init__(
        self,
        path: str,
        rank: Rank = -1,
        num_machines: int = 1,
        background: bool = False,
        partition_server_ranks: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
          - path : path to the folder containing checkpoints.
          - background: if True, will do prefetch and store in a background
                        process
        """

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
            self.prefetched: Dict[str, EntityPartitionType] = {}

        self.partition_client: Optional[PartitionClient] = None
        if partition_server_ranks is not None and len(partition_server_ranks) > 0:
            self.partition_client = PartitionClient(partition_server_ranks)

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

    def _version_ext(self, dirty: bool = False) -> str:
        version = self.checkpoint_version
        if dirty:
            version += 1
        return ('.%d' % version) if version > 0 else ''

    def _file_path(self, entity: EntityName, part: Partition) -> str:
        ext = self._version_ext((entity, part) in self.dirty)
        file_path = os.path.join(self.path, "%s_%d.pt%s" % (entity, part, ext))
        return file_path

    def write(
        self,
        entity: EntityName,
        part: Partition,
        embs: torch.FloatTensor,
        optim_state: OptimizerStateDict,
    ) -> None:
        self.dirty.add((entity, part))

        file_path = self._file_path(entity, part)

        if self.background:
            self._sync(file_path)

        if self.partition_client is not None:
            self.partition_client.store(entity, part, embs, optim_state)
        elif self.background:
            if file_path in self.prefetched:
                self.prefetched.pop(file_path)
            future_res = self.pool.apply_async(
                save_entity_partition, (file_path, embs, optim_state))
            self.outstanding[file_path] = future_res
        else:
            save_entity_partition(file_path, embs, optim_state)

    def read(
        self,
        entity: EntityName,
        part: Partition,
        *,
        force_dirty: bool = False,
    ) -> EntityPartitionType:
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
    ) -> Tuple[Optional[torch.FloatTensor], Optional[OptimizerStateDict]]:
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

    def write_metadata(
        self,
        config: ConfigSchema,
        model_state: ModuleStateDict,
        optim_state: OptimizerStateDict,
    ) -> None:
        ext = self._version_ext(True)
        file_path = os.path.join(self.path, "METADATA_0.pt%s" % ext)
        save_metadata(file_path, config, model_state, optim_state)

    def read_metadata(self) -> MetadataType:
        ext = self._version_ext(False)
        file_path = os.path.join(self.path, "METADATA_0.pt%s" % ext)
        return load_metadata(file_path)

    def maybe_read_metadata(
        self,
    ) -> Tuple[Optional[ConfigSchema], Optional[ModuleStateDict], Optional[OptimizerStateDict]]:
        try:
            return self.read_metadata()
        except FileNotFoundError:
            return None, None, None

    def _write_version_file(self, version: int) -> None:
        with open(os.path.join(self.path, VERSION_FILE), "wt") as tf:
            tf.write("%d" % version)
            tf.flush()
            os.fsync(tf.fileno())

    def write_new_version(self, config: ConfigSchema) -> None:
        if self.background:
            self._sync()
        new_ext = self._version_ext(True)
        if self.partition_client is not None:
            for entity, econf in config.entities.items():
                for part in range(self.rank, econf.num_partitions, self.num_machines):
                    vlog("Rank %d: getting %s %d" % (self.rank, entity, part))
                    embs, optim_state = \
                        self.partition_client.get(EntityName(entity), Partition(part))
                    vlog("Rank %d: saving %s %d to disk" % (self.rank, entity, part))
                    new_file_path = os.path.join(
                        self.path, "%s_%d.pt%s" % (entity, part, new_ext))
                    save_entity_partition(new_file_path, embs, optim_state)

    def switch_to_new_version(self) -> None:
        self.dirty.clear()
        self.checkpoint_version += 1
        if self.rank == 0:
            vlog("Rank 0: write version file")
            self._write_version_file(self.checkpoint_version)
            vlog("Rank 0: done")

    def remove_old_version(self, config: ConfigSchema) -> None:
        old_ext = '.%d' % (self.checkpoint_version - 1)
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                old_file_path = os.path.join(
                    self.path, "%s_%d.pt%s" % (entity, part, old_ext))
                vlog("%d os.remove %s" % (self.rank, old_file_path))
                if self.checkpoint_version > 1 or os.path.exists(old_file_path):
                    os.remove(old_file_path)

    def close(self) -> None:
        if self.background:
            self.pool.close()
            self.pool.join()

    def join(self) -> None:
        # FIXME: this whole join thing doesn't work with torch.distributed
        # can just get rid of it
        if self.partition_client is not None:
            self.partition_client.join()
