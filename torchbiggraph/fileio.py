#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import sys

import h5py
import torch
from torch_extensions.tensorlist.tensorlist import TensorList
from torch_extensions.rpc.rpc import (
    _serialize as torch_rpc_serialize,
    _deserialize as torch_rpc_deserialize,
)

from .entitylist import EntityList
from .util import log, vlog, create_workers, join_workers


def torch_load_shared(filename):
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


class EdgeReader(object):
    """Reads partitioned edgelists from disk, in the format
    created by edge_downloader.py.

    Edge lists are stored as hdf5 allowing partial reads (for multi-pass).

    Currently simple implementation but should eventually be multi-threaded /
    pipelined.
    """

    def __init__(self, path, index_base: int):
        if not os.path.isdir(path):
            raise RuntimeError("Invalid edge dir: %s" % path)
        self.path = path
        self.index_base = index_base

    def read(self, lhsP=0, rhsP=0, i=0, N=1):
        file_path = os.path.join(
            self.path,
            "edges_%d_%d.h5" % (lhsP + self.index_base, rhsP + self.index_base),
        )
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

            if self.index_base != 0:
                lhs -= self.index_base
                rhs -= self.index_base
                rel -= self.index_base

            return (EntityList(lhs, lhsd),
                    EntityList(rhs, rhsd),
                    rel)

    def read_dynamic(self, f, key, begin, end):
        offsets_field = '%s_offsets' % key
        data_field = '%s_data' % key
        if offsets_field in f and data_field in f:
            offsets = torch.from_numpy(f[offsets_field][begin:end + 1]).long()
            data = torch.from_numpy(f[data_field][offsets[0]:offsets[-1]]).long()

            if self.index_base != 0:
                data -= self.index_base

            # careful! as of Pytorch 0.4, offsets[0] is a 0d tensor view, so
            # offsets -= offsets[0] will give the wrong result
            offsets -= offsets[0].item()

            return TensorList(offsets, data)
        else:
            # Empty tensor_list representation
            return TensorList(
                torch.zeros(end - begin + 1).long(), torch.Tensor([])
            )

def _torch_save(data, file_path):
    vlog("Saving to %s" % file_path)
    with open(file_path, 'wb') as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())

def process_io_command(cmd, path, data):
    if cmd == 'save':
        _torch_save(data, path)
        vlog("Worker thread done save of %s" % path)
        return (cmd, path, None)
    elif cmd == 'load':
        vlog("Worker thread starting load of %s" % path)
        res = torch_load_shared(path)
        vlog("Worker thread done load of %s" % path)
        return (cmd, path, res)
    elif cmd == 'event':
        vlog("Processing event...")
        return (cmd, None, None)


VERSION_FILE = 'CHECKPOINT_VERSION'


class PartitionClient(object):
    """A wrapper around ParameterServerClient that knows how to read and write
    partitions (i.e. pairs (embs, optim_state)) to the parameter servers.
    """

    def __init__(self, server_ranks):
        from .parameterserver import ParameterServerClient
        self._clients = [ParameterServerClient(rank) for rank in server_ranks]

    def store(self, entity, part, data):
        embs, optim_state = data
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        client.store(key + "__embs", embs.data)
        client.store(key + "__optim", torch_rpc_serialize(optim_state))

    def get(self, entity, part):
        client = self._clients[part % len(self._clients)]
        key = "%s_%s" % (entity, part)
        embs = client.get(key + "__embs", shared=True)
        optim_state = torch_rpc_deserialize(client.get(key + "__optim"))
        return (embs, optim_state)

    def join(self):
        for client in self._clients:
            client.join()


class CheckpointManager(object):
    """Reads and writes checkpoint data to/from disk.

    Checkpoints are saved via torch.save with no introspection into the
    contents of the data.

    Swapped-out partitions are saved to disk with a temporary extension, and when
    `commit()` is called, that extension is marked as the 'checkpoint' extension
    via the CHECKPOINT_VERSION file. This scheme is chosen to work with shared
    filesystems (specifically, Gluster) which guarantee close/open data consistency
    but no metadata consistency (so os.rename is out).

    """

    def __init__(self, path, rank=-1, num_machines=1, background=False, partition_server_ranks=None, index_base=None):
        """
        Args:
          - path : path to the folder containing checkpoints.
          - background: if True, will do prefetch and store in a background
                        process
        """

        self.path = path
        self.dirty = set()
        self.rank = rank
        self.num_machines = num_machines
        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(os.path.join(path, VERSION_FILE)):
                self._write_version_file(0)

        # FIXME: there's a slight danger here, say that a multi-machine job fails
        # after a few versions, and then it reruns but one of the write_version=False
        # machines has cached the metadata and thinks it doesn't exist, then it
        # will expect checkpoint_version=0 and fail.
        if os.path.exists(os.path.join(self.path, VERSION_FILE)):
            with open(os.path.join(self.path, VERSION_FILE)) as f:
                vstr = f.read()
                self.checkpoint_version = int(vstr) if vstr != '' else 0
        else:
            self.checkpoint_version = 0

        self.background = background
        if self.background:
            io_worker, qin, qout = create_workers(
                N=1,
                F=lambda _rank, _args, cmd, data, path: process_io_command(
                    cmd, data, path),
                args=None
            )
            self.io_worker = io_worker[0]
            self.qin = qin[0]
            self.qout = qout[0]
            self.outstanding = set()
            self.prefetched = {}
            self.events_outstanding = 0

        if partition_server_ranks is not None and len(partition_server_ranks) > 0:
            self.partition_client = PartitionClient(partition_server_ranks)
        else:
            self.partition_client = None

        version_ext = ".%d" % self.checkpoint_version if self.checkpoint_version > 0 else ''
        if index_base is not None:
            self.index_base = index_base
        elif os.path.exists(os.path.join(self.path, "METADATA_0.pt%s" % version_ext)):
            self.index_base = 0
        elif os.path.exists(os.path.join(self.path, "METADATA_1.pt%s" % version_ext)):
            self.index_base = 1
        else:
            self.index_base = 0

    def record_event(self):
        assert self.background
        self.events_outstanding += 1
        self.qin.put(('event', None, None))

    def wait_events(self):
        self._sync(None, events=True)

    def _sync(self, sync_path, events=False):
        assert self.background
        vlog("CheckpointManager=>_sync( %s %d )" % (sync_path, events))
        vlog("outstanding= %s" % self.outstanding)
        while ((events and self.events_outstanding > 0) or
               (sync_path is not None and sync_path in self.outstanding) or
               not self.qout.empty()):

            res = self.qout.get(True)
            if isinstance(res, BaseException):
                print("Error in background I/O thread:", file=sys.stderr)
                sys.stderr.flush()
                raise res
            cmd, path, data = res
            if cmd == 'event':
                self.events_outstanding -= 1
                continue

            self.outstanding.remove(path)
            if cmd == 'load':
                log("Setting prefetched %s; %d outstanding" %
                    (path, len(self.outstanding)))
                self.prefetched[path] = data

    def _version_ext(self, dirty=False):
        version = self.checkpoint_version
        if dirty:
            version += 1
        return ('.%d' % version) if version > 0 else ''

    def _file_path(self, entity, part):
        ext = self._version_ext((entity, part) in self.dirty)
        file_path = os.path.join(
            self.path, "%s_%d.pt%s" % (entity, part + self.index_base, ext)
        )

        return file_path

    def write(self, entity, data, part=0):
        self.dirty.add((entity, part))

        file_path = self._file_path(entity, part)

        if self.background:
            self._sync(file_path)

        if self.partition_client is not None and len(data) == 2:
            self.partition_client.store(entity, part, data)
        elif self.background:
            if file_path in self.prefetched:
                self.prefetched.pop(file_path)
            self.qin.put(('save', file_path, data))
            self.outstanding.add(file_path)
        else:
            _torch_save(data, file_path)
            vlog('fileio.py: done writing path = %s' % file_path)

    def read(self, entity, part=0, strict=False, force_dirty=False):
        # if counter > 1, we are inside a pass. Otherwise, we are doing
        # evals or just finished an epoch so ".new" files do not exist.
        if force_dirty:
            self.dirty.add((entity, part))

        strict = strict or (entity, part) in self.dirty

        file_path = self._file_path(entity, part)
        if (entity, part) in self.dirty and self.partition_client is not None:
            res = self.partition_client.get(entity, part)
            assert not (res is None and strict)
            return res
        elif self.background:
            self._sync(file_path)
            if file_path in self.prefetched:
                return self.prefetched.pop(file_path)

        vlog('CheckPointManager.read=>file_path: ' + file_path)
        if strict or os.path.exists(file_path):
            # we know there should be a file, we may just have to wait for it
            # to sync on gfsai (for distributed)
            # note, the outer function is retryable, so this doesn't have to be
            return torch_load_shared(file_path)
        else:
            return None

    def prefetch(self, entity, part=0, strict=False):
        if not self.background:
            return

        file_path = self._file_path(entity, part)
        if file_path in self.outstanding or file_path in self.prefetched:
            vlog("Bailing from prefetch of %s" % file_path)
            return
        if os.path.exists(file_path):
            self.qin.put(('load', file_path, None))
            self.outstanding.add(file_path)
        assert not strict, "Did not find a checkpoint at %s" % file_path
        return None

    def write_metadata(self, config, edge_path_count, pazz, model_state, optim_state):
        self.write('METADATA',
                   (config.to_dict(), edge_path_count, pazz, model_state, optim_state))

    def read_metadata(self, strict=False):
        data = self.read('METADATA', strict=strict)
        if data is not None:
            return data
        else:
            return None, 0, -1, None, None

    def _write_version_file(self, version):
        with open(os.path.join(self.path, VERSION_FILE), 'w') as f:
            f.write(str(version))
            f.flush()
            os.fsync(f.fileno())

    def commit(self, config):
        if self.background:
            self.record_event()
            self.wait_events()
        self.dirty.clear()
        self.checkpoint_version += 1
        new_ext = self._version_ext(False)
        assert self.rank >= 0
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                if self.partition_client is not None:
                    vlog("Rank %d: get %s %d" % (self.rank, entity, part))
                    data = self.partition_client.get(entity, part)
                    vlog("Rank %d saving to disk" % self.rank)
                    new_file_path = os.path.join(
                        self.path, "%s_%d.pt%s" % (entity, part + self.index_base, new_ext)
                    )
                    _torch_save(data, new_file_path)

        if self.rank == 0:
            vlog("Rank 0: write version file")
            self._write_version_file(self.checkpoint_version)
            vlog("Rank 0: done")

    def remove_old_version(self, config):
        old_ext = '.%d' % (self.checkpoint_version - 1)
        for entity, econf in config.entities.items():
            for part in range(self.rank, econf.num_partitions, self.num_machines):
                old_file_path = os.path.join(
                    self.path, "%s_%d.pt%s" % (entity, part + self.index_base, old_ext)
                )
                vlog("%d os.remove %s" % (self.rank, old_file_path))
                if self.checkpoint_version > 1 or os.path.exists(old_file_path):
                    os.remove(old_file_path)

    def close(self):
        if self.background:
            join_workers([self.io_worker], [self.qin], [self.qout])

    def join(self):
        # FIXME: this whole join thing doesn't work with torch.distributed
        # can just get rid of it
        if self.partition_client is not None:
            self.partition_client.join()
