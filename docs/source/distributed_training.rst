.. _distributed-training:

Distributed mode
================

PBG can perform training across multiple machines which communicate over a network,
in order to reduce training time on large graphs. Distributed training is able to
concurrently utilize larger computing resources, as well as to keep the entire model
stored in memory across all machines, avoiding the need to swap it to disk.
On each machine, the training is further parallelized across multiple subprocesses.

Setup
-----

In order to perform distributed training, the configuration file must first be updated to contain
the specification of the desired distributed setup. If training should be carried out on :math:`N`
machines, then the ``num_machines`` key in the config must be set to that value. In addition, the
``distributed_init_method`` must describe a way for the trainers to discover each other and set up
their communication. All valid values for the ``init_method`` argument of :func:`torch.distributed.init_process_group`
are accepted here. Usually this will be a path to a shared network filesystem or the network address of one of the machines.
See the `PyTorch docs <https://pytorch.org/docs/stable/distributed.html#initialization>`_ for more information and a complete reference.

To launch distributed training, call :samp:`torchbiggraph_train --rank {rank} {config.py}` on each machine,
with ``rank`` replaced by an integer between 0 and :math:`N-1` (inclusive), different for each machine.
Each machine must have PBG installed and have a copy of the config file.

In some uncommon circumstances, one may want to store the embeddings on different machines than the ones that
are performing training. In that case, one would set ``num_partition_servers`` to a positive value and manually
launch some instances of ``torchbiggraph_partitionserver`` as well. See below for more information on this.

.. tip:: A good default setting is to set ``num_machines`` to *half* the number of partitions (see
  below why) and leave ``num_partition_servers`` unset.

Once all these commands are started, no more manual intervention is needed.

.. warning::

  Unpartitioned entity types should not be used with distributed training. While
  the embeddings of partitioned entity types are only in use on one machine at a
  time and are swapped between machines as needed, the embeddings of unpartitioned
  entity types are communicated asynchronously through a poorly-optimized parameter
  server which was designed for sharing relation parameters, which are small. It
  cannot support synchronizing large amounts of parameters, e.g. an unpartitioned
  entity type with more than 1000 entities. In that case, the quality of the
  unpartitioned embeddings will likely be very poor.

Communication protocols
-----------------------

Distributed training requires the machines to coordinate and communicate in various ways for different purposes.
These tasks are:

- synchronizing which trainer is operating on which bucket, assigning them so that there are no conflicts
- passing the embeddings of an entity partition from one trainer to the next one when needed (as this is data that is only
  accessed by one trainer at a time)
- sharing parameters that all trainers need access to simultaneously, by collecting and redistributing the updates to them.

Each of these is implemented by a separate "protocol", and each trainer takes part in some or all of them by launching
subprocesses that act as clients or servers for the different protocols. These protocols are explained below to provide insight into the system.

Synchronizing bucket access
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PBG parallelizes training across multiple machines by having them all operate simultaneously on disjoint buckets
(i.e., buckets that don't have any partition in common). Therefore, each partition is in use by up to one machine at a
time, and each machine uses up to two partitions (the only exception is for buckets "on the diagonal", that have the same
left- and right-hand side partition). This means that the number of buckets one can simultaneously train on is about half
the total number of partitions.

The way the machines agree on which one gets to operate on what bucket is through a "**lock server**". The server is
implicitly started by the trainer of rank 0. All other machines connect to it as clients, ask for a new bucket to operate
on (when they need one), get a bucket assigned from the server (or none, if all buckets have already been trained on or
are "locked" because their partitions are in use by another trainer), train on it, then report it as done and repeat. The
lock server tries to optimize I/O by preferring, when a trainer asks for a bucket, to assign one that has as many partitions
in common with the previous bucket that the trainer trained on, so that these partitions can be kept in memory rather than
having to be unloaded and reloaded.

Exchanging partition embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a trainer starts operating on a bucket it needs access to the embeddings of all entities (of all types) that belong to
either the left- or the right-hand side partition of the bucket. The "locking" mechanism of the lock server ensures that at most
one trainer is operating on a partition at any given time. This doesn't hold for unpartitioned entity types, which are shared
among all trainers; see below. Thus each trainer has exclusive hold of the partitions it's training on.

Once a trainer starts working on a new bucket it needs to acquire the embeddings of its partitions, and once it's done it needs to
release them and make them available, in their updated version, to the next trainer that needs them. In order to do this, there's a
system of so-called "**partition servers**" that store the embeddings, provide them upon request to the trainers who need them,
receive back the updated embedding and store it.

This service is optional, and is disabled when ``num_partition_servers`` is set to zero. In that case the trainers "send"
each other the embeddings simply by writing them to the checkpoint directory (which should reside on a shared disk) and
then fetching them back from there.

When this system is enabled, it can operate in two modes. The simplest mode is triggered when ``num_partition_servers``
is -1 (the default): in that case all trainers spawn a local process that acts as a partition server. If, on the other hand,
``num_partition_servers`` is a positive value then the trainers will not spawn any process, but will instead connect to
the partition servers that the user must have provisioned manually by launching the ``torchbiggraph_partitionserver``
command on the appropriate number of machines.

Updating shared parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some parameters of the model need to be used by all trainers at the same time (this includes the operator weights,
the global embeddings of each entity type, the embeddings of the unpartitioned entities). These are parameters that
don't depend on what bucket the trainer is operating on, and therefore are always present on all trainers (as opposed
to the entity embeddings, which are loaded and unloaded as needed). These parameters are synchronized using a series
of "**parameter servers**". Each trainer starts a local parameter server (in a separate subprocess) and connects to
all other parameter servers. Each parameter that is shared between trainers is then stored in a parameter server (possibly
sharded across several of them, if too large). Each trainer also has a loop (also in a separate subprocess) which, at regular intervals, goes over each shared parameter,
computes the difference between its current local value and the value it had when it was last synced with the
server where the parameter is hosted and sends that delta to that server. The server, in turn, accumulates all the deltas
it receives from all trainers, updates the value of the parameter and sends this new value back to the trainers. The parameter server
performs throttling to 100 updates/s or 1GB/s, in order to prevent the parameter server from starving the other
communication.

.. todo:: Mention ``distributed_tree_init_order``.
