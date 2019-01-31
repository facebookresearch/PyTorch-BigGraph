#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import time
import traceback

import torch
import torch.multiprocessing as mp

from .util import log, init_process_group


################################################################################
# Generic parameter server
################################################################################


# FIXME! This will be slow af
def _tostring(t):
    return "".join(chr(x) for x in t)


def _fromstring(s):
    return torch.CharTensor([ord(x) for x in s])


STORE_CMD = 1
GET_CMD = 2
JOIN_CMD = 3
SWAP_CMD = 4

_tensor_types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.ByteTensor,
    torch.IntTensor,
    torch.LongTensor,
]


_tensor_type_idx = {t().type(): i for i, t in enumerate(_tensor_types)}


class ParameterServer(object):
    """
    A simple parameter server. Clients can store tensors, accumulate, and
    get tensors by string key. Operations on the parameter server are globally
    synchronous.

    FIXME: torch_extensions.rpc should be fixed to not require torch.serialization,
    then most of this code can be removed.
    FIXME: torch.distributed.recv should not require you to provide the
    tensor to write to; the type and size should be sent in the header.
    That would simplify this code a lot.
    """

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.parameters = {}

    def start(self):
        join_count = 0
        while True:
            # 1. receive the command
            cmd_buffer = torch.LongTensor(6).fill_(-1)
            rank = torch.distributed.recv(cmd_buffer)
            cmd = cmd_buffer[0]

            if cmd == STORE_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_store(rank, key,
                                  cmd_buffer[2].item(),
                                  cmd_buffer[3].item(),
                                  cmd_buffer[4].item(),
                                  cmd_buffer[5].item())
            elif cmd == GET_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_get(rank, key, cmd_buffer[2].item())
            elif cmd == SWAP_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_store(rank, key,
                                  cmd_buffer[2].item(),
                                  cmd_buffer[3].item(),
                                  cmd_buffer[4].item(),
                                  cmd_buffer[5].item())
                self.handle_get(rank, key, False)
            elif cmd == JOIN_CMD:
                join_count += 1
                if join_count == self.num_clients:
                    for r in range(self.num_clients):
                        # after sending the join cmd,
                        # each client waits on this ack to know everyone is done
                        # and it's safe to exit
                        torch.distributed.send(torch.Tensor(1).fill_(0), dst=r)
                    break
            else:
                raise RuntimeError("Command is unknown value %d from rank %d."
                        % (cmd, rank))

    def _recv_key(self, rank, keylen):
        """Receive a string tensor key from a client node."""
        key_buffer = torch.CharTensor(keylen).fill_(0)
        torch.distributed.recv(key_buffer, src=rank)
        return _tostring(key_buffer)

    def handle_store(self, rank, key, ndim, accum, overwrite, ttype):

        if ndim == -1:
            assert key in self.parameters
            size = self.parameters[key].size()
        else:
            size = torch.LongTensor(ndim)
            torch.distributed.recv(size, src=rank)
            size = size.tolist()
        tensor_type = _tensor_types[ttype]
        data = tensor_type(*size)
        torch.distributed.recv(data, src=rank)

        if accum:
            self.parameters[key] += data
        elif (key not in self.parameters) or overwrite:
            self.parameters[key] = data

    def handle_get(self, rank, key, send_size):

        if key not in self.parameters:
            assert send_size, "Key %s not found" % key
            torch.distributed.send(torch.LongTensor([-1, -1]), rank)
            return

        data = self.parameters[key]
        if send_size:
            type_idx = _tensor_type_idx[data.type()]
            torch.distributed.send(
                torch.LongTensor([data.ndimension(), type_idx]), rank)
            torch.distributed.send(torch.LongTensor(list(data.size())), rank)

        torch.distributed.send(data, dst=rank)


class ParameterServerClient(object):
    """Client for ParameterServer.
    Supports store, accumulate, swap, swap-accumulate, and get operations."""

    def __init__(self, server_rank):
        self.server_rank = server_rank

    def store(self, key, src, accum=False, overwrite=True):
        """Store or accumulate a tensor on the server.
        """
        cmd_rpc = torch.LongTensor([STORE_CMD,
                                    len(key),
                                    -1 if accum else src.ndimension(),
                                    int(accum),
                                    int(overwrite),
                                    _tensor_type_idx[src.type()]])
        torch.distributed.send(cmd_rpc, self.server_rank)
        torch.distributed.send(_fromstring(key), self.server_rank)
        if not accum:
            torch.distributed.send(torch.LongTensor(list(src.size())),
                                   self.server_rank)
        torch.distributed.send(src, self.server_rank)

    def get(self, key, dst=None, shared=False):
        """Get a tensor from the server.
        """
        cmd_rpc = torch.LongTensor([GET_CMD, len(key), dst is None, 0, 0, 0])
        torch.distributed.send(cmd_rpc, self.server_rank)
        torch.distributed.send(_fromstring(key), self.server_rank)
        if dst is None:
            meta = torch.LongTensor(2).fill_(-1)
            torch.distributed.recv(meta, src=self.server_rank)
            ndim, ttype = meta
            if ndim.item() == -1:
                return None
            size = torch.LongTensor(ndim.item()).fill_(-1)
            torch.distributed.recv(size, src=self.server_rank)
            tensor_type = _tensor_types[ttype.item()]
            if shared:
                dst_storage = tensor_type().storage_type()._new_shared(size.prod())
                dst = tensor_type(dst_storage).view(*size.tolist())
            else:
                dst = tensor_type(*size.tolist())
        torch.distributed.recv(dst, src=self.server_rank)
        return dst

    def swap(self, key, src, dst=None, accum=False, overwrite=False):
        """Store or accumulate a tensor on the server,
        and then get its current value.
        """
        if dst is None:
            dst = torch.zeros_like(src)

        # tic = time.time()
        cmd_rpc = torch.LongTensor([SWAP_CMD,
                                    len(key),
                                    -1 if accum else src.ndimension(),
                                    int(accum),
                                    int(overwrite),
                                    _tensor_type_idx[src.type()]])
        torch.distributed.send(cmd_rpc, self.server_rank)
        torch.distributed.send(_fromstring(key), self.server_rank)
        if not accum:
            torch.distributed.send(torch.LongTensor(list(src.size())),
                                   self.server_rank)
        torch.distributed.send(src, self.server_rank)
        torch.distributed.recv(dst, src=self.server_rank)
        # log("Swapped %d bytes to %d in %g s" %
        #     (src.nelement(), self.server_rank, time.time() - tic))

    def join(self):
        """All clients should call join at the end, which will allow the server
        to exit.
        """

        cmd_rpc = torch.LongTensor([JOIN_CMD, 0, 0, 0, 0, 0])
        torch.distributed.send(cmd_rpc, self.server_rank)
        ack = torch.Tensor(1)
        torch.distributed.recv(ack, src=self.server_rank)


class GradientParameterServerClient(object):
    """We keep track of the last pull of each tensor from the server, and then when
    a push is requested, we accumulate the difference between the pulled tensor
    and the current version
    """

    def __init__(self, server_rank):
        self._client = ParameterServerClient(server_rank)
        self._cache = {}

    def push(self, key, tensor):
        # if they tensor is cached, accumulate the difference.
        # otherwise, send the tensor to the server.
        if key in self._cache:
            diff = tensor - self._cache[key]
            self._client.store(key, diff, accum=True)
            self._cache[key] += diff
        else:
            self._cache[key] = tensor.clone()
            # all the clients race to set the initial value of the tensor, then
            # every clients just uses that one
            self._client.store(key, self._cache[key], overwrite=False)

    def pull(self, key, dst):
        self._client.get(key, dst)
        if key in self._cache:
            self._cache[key].copy_(dst)
        else:
            self._cache[key] = dst.clone()
        return dst

    def update(self, key, tensor):
        if key in self._cache:
            diff = tensor - self._cache[key]
            self._client.swap(key, diff, self._cache[key], accum=True)
            tensor.copy_(self._cache[key])
        else:
            self._cache[key] = tensor.clone()
            # all the clients race to set the initial value of the tensor, then
            # every clients just uses that one
            self._client.swap(key, self._cache[key], self._cache[key],
                              overwrite=False)
            tensor.copy_(self._cache[key])

    def join(self):
        self._client.join()

################################################################################
# Project-specific additions
################################################################################


MIN_BYTES_TO_SHARD = 1e7  # only shard parameters above 10MB

def _client_thread_loop(process_group_params,
                        client_rank,
                        all_server_ranks,
                        q,
                        errq,
                        max_bandwidth=1e8,
                        min_sleep_time=0.01):
    try:

        init_process_group(rank=client_rank,
                           **process_group_params)

        params = {}
        clients = [GradientParameterServerClient(server_rank) for
                   server_rank in all_server_ranks]
        log_time, log_rounds, log_bytes = time.time(), 0, 0

        # thread loop:
        # 1. check for a command from the main process
        # 2. update (push and pull) each parameter in my list of parameters
        # 3. if we're going to fast, sleep for a while
        while True:
            tic = time.time()
            bytes_transferred = 0
            try:
                data = q.get(timeout=0.01)
                cmd, args = data
                if cmd == "params":
                    params[args[0]] = args[1]
                    log_time, log_rounds, log_bytes = time.time(), 0, 0
                elif cmd == "join":
                    for client in clients:
                        client.join()
                    break
            except queue.Empty:
                pass

            for k, v in params.items():
                bytes_transferred += v.nelement() * 4  # assume float
                if v.nelement() * 4 > MIN_BYTES_TO_SHARD:
                    chunks = v.chunk(len(clients), dim=0)
                    for client, chunk in zip(clients, chunks):
                        client.update(k, chunk)
                else:
                    client_idx = hash(k) % len(clients)
                    clients[client_idx].update(k, v)

            log_bytes += bytes_transferred
            log_rounds += 1
            log_delta = time.time() - log_time
            if params and log_delta > 60:
                log("Parameter client synced %d rounds %g GB in %g s ( %g s/round , %g GB/s)" %
                    (log_rounds, log_bytes / 1e9, log_delta,
                     log_delta / log_rounds, log_bytes / log_delta / 1e9))
                log_time, log_rounds, log_bytes = time.time(), 0, 0

            comm_time = time.time() - tic
            sleep_time = max(bytes_transferred / max_bandwidth - comm_time,
                             min_sleep_time)
            time.sleep(sleep_time)

    except BaseException as e:
        traceback.print_exc()
        errq.put(e)
        raise


class GradientParameterServerClientThread(object):
    """Wrapper object that creates a thread, that pulls parameters and pushes
    gradients, in a loop.
    """

    def __init__(self, process_group_params, client_rank, all_server_ranks):
        self.q = mp.Queue()
        self.errq = mp.Queue()
        self.p = mp.Process(
            target=_client_thread_loop,
            args=(process_group_params, client_rank, all_server_ranks, self.q, self.errq)
        )
        self.p.daemon = True
        self.p.start()

    def set_param(self, k, v):
        self.check()
        self.q.put(('params', (k, v)))

    def check(self):
        if not self.errq.empty():
            raise self.errq.get()

    def join(self):
        self.check()
        self.q.put(('join', None))
        self.check()
        self.p.join()


def _start_parameter_server(process_group_params, rank, num_clients):
    init_process_group(rank=rank,
                       **process_group_params)

    ps = ParameterServer(num_clients)
    ps.start()


def setup_parameter_server(server_rank,
                           num_clients,
                           world_size=None,
                           init_method=None,
                           groups=None):
    process_group_params = {
        'world_size': world_size,
        'init_method': init_method,
        'groups': groups
    }

    # set up the parameter server on rank 0, but as a separate node
    # with MPI rank numMachines
    p_server = mp.Process(target=_start_parameter_server,
                          args=(process_group_params, server_rank, num_clients)
                          )
    p_server.daemon = True
    p_server.start()

def setup_parameter_server_thread(client_rank,
                                  server_rank,
                                  all_server_ranks,
                                  num_clients,
                                  world_size=None,
                                  init_method=None,
                                  groups=None):

    process_group_params = {
        'world_size': world_size,
        'init_method': init_method,
        'groups': groups
    }

    setup_parameter_server(server_rank,
                           num_clients,
                           **process_group_params)

    client = GradientParameterServerClientThread(
        process_group_params, client_rank, all_server_ranks)

    return client
