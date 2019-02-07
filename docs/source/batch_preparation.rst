Batch preparation
=================

This chapter presents how the training data is prepared and organized in batches
before the loss is :ref:`calculated <loss-calculation>` and :ref:`optimized <optimizers>`
on each of them.

Passes
------

Training proceeds by iterating over the edges, through various nested loops. The
outermost one walks through so-called **epochs**. Each epoch is independent and
essentially equivalent to every other one. Their goal is to repeat the inner loop
until convergence. Each epoch visits all the edges exactly once. The number of
epochs is specified in the ``num_epochs`` configuration parameter.

The edges are partitioned into **edge sets** (one for each directory of the ``edge_paths``
configuration key) and, within each epoch, the edge sets are traversed in order.

When iterating over an edge set, each of its buckets is first divided into
equally sized **chunks**: each chunk spans a contiguous interval of edges (in the
order they are stored in the files) and the number of chunks can be tweaked
using the ``num_edge_chunks`` configuration key. The training first operates
on the all the first chunks of all buckets, then on all of their second chunks,
and so on.

Next, the algorithm iterates over the **buckets**. The order in which buckets are
processed depends on the value of the ``bucket_order`` configuration key. In
addition to a random permutation, there are methods that try to have successive
buckets share a common partition: this allows for that partition to be reused,
thus allowing it to be kept in memory rather than being unloaded and another one
getting loaded in its place.

.. note::
    In :ref:`distributed mode <distributed-training>`, the various trainer processes
    operate on the buckets at the same time, thus the iteration is managed differently.

Once the trainer has fixed a given chunk and a certain bucket, its edges are
finally loaded from disk. When
:ref:`evaluating during training <evaluation-during-training>`, a subset of these
edges is withheld (such subset is the same for all epochs). The remaining edges
are immediately uniformly shuffled, then split into equal parts, and one part is
given to each **"Hogwild!" worker** for the training to proceed in parallel.
The number of such workers is determined by the ``workers`` parameter.

The way each worker trains on its set of edges depends on whether
:ref:`dynamic relations <dynamic-relations>` are in use. The simplest scenario is if
they are, in which case the edges are split into contiguous **batches** (each one having
the size specified in the ``batch_size`` configuration key, except possibly the last
one which could be smaller). Training is then performed on that batch before moving
on to the next one.

When dynamic relations are not in use, however, the loss can only be computed on
a set of edges that are all of the same type. Thus the worker first randomly
samples a relation type, with probability proportional to the number of edges
of that type that are left in the pool. It then takes the first ``batch_size`` relations of
that type (or fewer, if not enough of them are left), removes them from the pool and
performs training on them.

.. _distributed-training:

Distributed mode
================

Talk about what invocations need to be done (``num_machines`` trainers plus, in
case, ``num_partition_servers`` partition servers)

Talk about what processes they each spawn, how they communicate (using ``torch.distributed``, or queues, or the filesystem, ...)

Talk about the different groups of processes (lockserver, barriers, partition and parameter servers, ...)
