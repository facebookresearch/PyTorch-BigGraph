.. _io-format:

I/O format
==========

Entity and relation types
-------------------------

The list of entity types (each identified by a string), plus some information
about each of them, is given in the ``entities`` dictionary in the configuration file.
The list of relation types (each identified by its index in that list), plus
some data like what their left- and right-hand side entity types are, is in the
``relations`` key of the configuration file.

Entities
--------

The only information that needs to be provided about entities is how many there
are in each entity type's partition. This is done by putting a file named :file:`entity_count_{type}_{part}.txt` for each entity type identified
by ``type`` and each partition ``part`` in the directory specified by the ``entity_path`` config parameter. These files must contain a single
integer (as text), which is the number of entities in that partition. The directory where all these
files reside must be specified as the ``entity_path`` key of the configuration file.

It is possible to provide an initial value for the embeddings, by specifying a
value for the ``init_path`` configuration key, which is the name of a directory that
contains files in a format similar to the output format detailed in
:ref:`output-format` (possibly without the optimizer state dicts).

If no initial value is provided, it will be auto-generated, with each dimension
sampled from the centered normal distribution whose standard deviation can be
configured using the ``init_scale`` configuration key. For performance reasons
the samples of all the entities of a certain type will not be independent.

Edges
-----

For each bucket there must be a file that stores all the edges that fall in that
bucket, of all relation types. This means that such a file is only identified by
two integers, the partitions of its left- and right-hand side entities. It must
be named :file:`edges_{lhs}_{rhs}.h5` (where ``lhs`` and ``rhs`` are the above
integers), it must be a `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ file
containing three one-dimensional datasets of the same length, called ``rel``,
``lhs`` and ``rhs``. The elements in the :math:`i`-th positions in each of them
define the :math:`i`-th edge: ``rel`` identifies the relation type (and thus the
left- and right-hand side entity types), ``lhs`` and ``rhs`` given the indices
of the left- and right-hand side entities within their respective partitions.

To ease future updates to this format, each file must contain the format version
in the ``format_version`` attribute of the top-level group. The current version is 1.

If an entity type is unpartitioned (that is, all its entities belong to the
same partition), then the edges incident to these entities must still be
uniformly spread across all buckets.

These files, for all buckets, must be stored in the same directory, which must
be passed as the ``edge_paths`` configuration key. That key can actually contain
a list of paths, each pointing to a directory of the format described above: in
that case the graph will contain the union of all their edges.

.. _output-format:

Checkpoint
----------

The training's checkpoints are also its output, and they are written to the directory
given as the ``checkpoint_path`` parameter in the configuration. Checkpoints are identified
by successive positive integers, starting from 1, and all the files belonging to
a certain checkpoint have an extra component :file:`.v{version}` between their name and extension
(e.g., :file:`{something}.v42.h5` for version 42).

The latest complete checkpoint version is stored in an additional file in the same directory, called
:file:`checkpoint_version.txt`, which contains a single integer number, the current version.

Each checkpoint contains a JSON dump of the config that was used to produce it stored in the :file:`config.json` file.

After a new checkpoint version is saved, the previous one will automatically be
deleted. In order to periodically preserve some of these versions, set the
``checkpoint_preservation_interval`` config flag to the desired period (expressed
in number of epochs).

Model parameters
^^^^^^^^^^^^^^^^

The model parameters are stored in a file named :file:`model.h5`, which is a HDF5 file containing
one dataset for each parameter, all of which are located within the ``model`` group. Currently, the
parameters that are provided are:

- :samp:`model/relations/{idx}/operator/{side}/{param}` with the parameters of each relation's operator.
- :samp:`model/entities/{type}/global_embedding` with the per-entity type global embedding.

Each of these datasets also contains, in the ``state_dict_key`` attribute, the key it was stored inside the
model state dict. An additional dataset may exist, ``optimizer/state_dict``, which contains the binary blob
(obtained through :func:`torch.save`) of the state dict of the model's optimizer.

Finally, the top-level group of the file contains a few attributes with additional metadata. This mainly
includes the format version, a JSON-dump of the config and some information about the iteration that produced
the checkpoint.

Embeddings
^^^^^^^^^^

Then, for each entity type and each of its partitions, there is a file
:file:`embeddings_{type}_{part}.h5` (where ``type`` is the type's name and ``part``
is the 0-based index of the partition), which is a HDF5 file with two datasets.
One two-dimensional dataset, called ``embeddings``, contains the embeddings of
the entities, with the first dimension being the number of entities and the
second being the dimension of the embedding.

Just like for the model parameters file, the optimizer state dict and additional metadata is also included.

HDFS Format
^^^^^^^^^^

Include the prefix ``hdfs://`` in entities, edges and checkpoint paths when running in distributed hdfs cluster.
