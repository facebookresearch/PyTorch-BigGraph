#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import sys
import traceback

import numpy as np
import torch
import torch.distributed as td


# FIXME: is it efficient to torch.save into a buf? It's going to have to copy
# all the tensors.
def _serialize(data):
    buf = io.BytesIO()
    torch.save(data, buf)
    data_bytes = buf.getvalue()
    # FIXME: how to properly copy bytes to ByteTensor?
    t = torch.from_numpy(np.frombuffer(data_bytes, dtype=np.uint8))
    return t


def _deserialize(t):
    data_bytes = t.numpy().tobytes()
    buf = io.BytesIO(data_bytes)
    return torch.load(buf)


def send(data, dst):
    """
    Sends an arbitrary torch-serializable object to a destination node.
    This is a blocking send, equivalent to `torch.distributed.send`.

    Args:
        data: An arbitrary torch-serializable object to be sent.
        dst: The rank of the destination node.
    """

    # FIXME: we've got to get rid of this two-pass nonsense for dynamically sized
    # send and receive.
    t = _serialize(data)
    sizet = torch.LongTensor([t.nelement()])
    td.send(sizet, dst)
    td.send(t, dst)


def recv(src=None):
    """
    Receives an arbitrary torch-serializable object from a source node.
    This is a blocking receive, `torch.distributed.recv`

    Args:
        src: The rank of the source node. If None, will receive from any rank.

    Returns:
        data: The data send from the source node.
        src: The rank of the source node.
    """
    sizet = torch.LongTensor(1)
    src = td.recv(sizet, src)
    t = torch.ByteTensor(sizet.item())
    td.recv(t, src)
    return _deserialize(t), src


_JOIN_KEY = "seU17sb9nwqDZhsH9AyW"


class Server(object):
    """Base class for an RPC server using `torch.distributed`.
    Users should subclass this class and add the server methods.

    Example:
        init_method = "file://myfile.tmp"
        num_clients = 1
        torch.distributed.init_process_group('gloo',
                                             init_method=init_method,
                                             world_size=num_clients + 1,
                                             rank=0)

        class MyServer(Server):
            def test_func(self, T, k=0):
                return ("the result is ", T + k)

        s = MyServer(num_clients)
        s.start()  # will block until all clients have called `join()`
    """

    def __init__(self, num_clients):
        """
        Args:
            num_clients: The number of clients that will call `join()` upon
                         completion.
        """
        self.num_clients = num_clients

    def start(self, groups=None):
        join_clients = []

        while True:
            rpc, src = recv()
            if rpc == _JOIN_KEY:
                join_clients += [src]
                if len(join_clients) == self.num_clients:
                    for client in join_clients:
                        # after sending the join cmd,
                        # each client waits on this ack to know everyone is done
                        # and it's safe to exit
                        send(_JOIN_KEY, client)
                    break
            else:
                F, args, kwargs = rpc
                try:
                    res = getattr(self, F)(*args, **kwargs)
                    send((False, res), src)
                except BaseException as e:
                    # should we print the exception on the server also?
                    # traceback.print_exc()
                    exc_str = traceback.format_exc()
                    send((True, (e, exc_str)), src)


class Client(object):
    """A client for connecting to a subclass of `rpc.Server`.

    Example:
        init_method = "file://myfile.tmp"
        num_clients = 1
        torch.distributed.init_process_group('gloo',
                                             init_method=init_method,
                                             world_size=num_clients + 1,
                                             rank=1)

        c = Client(MyServer, server_rank=0)

        print(c.test_func(torch.arange(0, 3), k=2))
        # ('the result is ', tensor([ 2,  3,  4]))

        c.join()
    """

    def __init__(self, server_class, server_rank):
        """
        Args:
            server_class: The class of the server object. This should be a
                          subclass of `rpc.Server`.
            server_rank: The rank of the node where the `rpc.Server` is running.
        """
        self.server_class = server_class
        self.server_rank = server_rank

    def __getattr__(self, name):
        if name not in dir(self.server_class):
            raise AttributeError(
                "%s has no attribute %s" % (self.server_class.__name__, name)
            )
        func = getattr(self.server_class, name)
        if not isinstance(func, type(lambda: 1)):  # FIXME
            raise TypeError("%s object is not callable" % (type(func)))

        def inner(*args, **kwargs):
            send((name, args, kwargs), self.server_rank)
            (is_exception, res), _src = recv(self.server_rank)
            if not is_exception:
                return res
            else:
                exc, exc_str = res
                print(exc_str, file=sys.stderr)
                raise exc

        return inner

    def join(self):
        """Should be called by each client upon completion, to ensure a clean exit.
        """
        send(_JOIN_KEY, self.server_rank)
        recv(self.server_rank)
