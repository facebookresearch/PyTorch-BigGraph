FAQ & Troubleshooting
=====================

Frequently Asked Questions
--------------------------

Undirected graphs
^^^^^^^^^^^^^^^^^

Edges in PBG's :ref:`data model <data-model>` are always interpreted as directed.
To operate on undirected data, it is often enough to replace each undirected edge
```a <-> b`` with two directed edges ``a -> b`` and ``b -> a``. In fact, even with data
that is already directed, it may be beneficial to artificially add a "reversed"
edge ``b ~> a`` for each original edge ``a -> b``, of a different relation type.
This is automatically done by PBG in the :ref:`dynamic relations mode <dynamic-relations>`.

Common issues
-------------

Bus error
^^^^^^^^^

Training might occasionally fail with a ``Bus error (core dumped)`` message, and
no traceback. This is often caused by the inability to allocate enough _shared_
memory, that is, memory that can be simultaneously accessed by multiple processes
(this is needed to perform training in parallel). Such an error is produced by
the kernel and it's hard to detect it in advance or catch it.

This may occur when running PBG inside a Docker container, as by default the
shared memory limit for them is rather small. `This PyTorch issue <https://github.com/pytorch/pytorch/issues/2244>`_
may provide some insight in how to address that. If this occurs on a Linux machine,
it may be fixed by increasing the size of the ``tmpfs`` mount on ``/dev/shm`` or
on ``/var/run/shm``.

It may also just be that the machine ran out of physical memory because the data
is too large. This is exactly the scenario PBG was designed for, and can be fixed
by increasing the number of partitions of the data.
