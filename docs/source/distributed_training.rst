.. _distributed-training:

Distributed mode
================

Talk about what invocations need to be done (``num_machines`` trainers plus, in
case, ``num_partition_servers`` partition servers)

Talk about what processes they each spawn, how they communicate (using ``torch.distributed``, or queues, or the filesystem, ...)

Talk about the different groups of processes (lockserver, barriers, partition and parameter servers, ...)
