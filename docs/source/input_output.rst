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
by ``type`` and each partition ``part`` in the directory specified by the ``entity_path`` config parameter. These files must contain an ASCII-encoded
integer, which is the number of entities in that partition.

It is possible to provide an initial value for the embeddings, by specifying a
value for the ``init_path`` configuration key, which is the name of a directory that
contains files in a format similar to the output format detailed in
:ref:`output-format`: the :file:`METADATA_1` file can be omitted, the optimizer
state can be ``None`` and, optionally, one can also omit the :file:`checkpoint_version.txt`
file and avoid adding a version suffix to any file.

.. note::
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

.. note::
    When using featurized entities this format will be different.

.. note::
    If an entity type is unpartitioned (that is, all its entities belong to the
    same partition), then the edges incident to these entities must still be
    uniformly spread across all buckets.

These files, for all buckets, must be stored in the same directory, which must
be passed as the ``edge_paths`` configuration key. That key can actually contain
a list of paths, each pointing to a directory of the format described above: in
that case the graph will contain the union of all their edges.

.. note::
    When using dynamic relations there also needs to be an additional file,
    named :file:`dynamic_rel_count.txt`, in the ``entity_path`` directory.

.. _output-format:

Checkpoint
----------

The training's checkpoints are also its output, and they are written to the directory
given as the ``checkpoint_path`` parameter in the configuration. Checkpoints are identified
by successive positive integers, starting from 1, and all the files belonging to
a certain checkpoint have their names end with :file:`.{version}`.

Each checkpoint contains a JSON dump of the config that was used to produce it
stored in the :file:`config.json` file and a metadata file named :file:`model.h5`,
which is a HDF5 file containing the parameters of the model (minus the entity
embeddings) inside the ``model`` group, and the state of the model optimizer
inside the ``optimizer/state_dict`` dataset.

Then, for each entity type and each of its partitions, there is a file
:file:`embeddings_{type}_{part}.h5` (where ``type`` is the type's name and ``part``
is the 0-based index of the partition), which is a HDF5 file with two datasets.
One two-dimensional dataset, called ``embeddings``, contains the embeddings of
the entities, with the first dimension being the number of entities and the
second being the dimension of the embedding. A second binary one-dimensional
dataset is the pickled state dictionary of the optimizer for those entities.

An additional file in the same directory, called :file:`checkpoint_version.txt`,
contains the latest checkpoint version, as an ASCII-encoded decimal number.
While the metadata files are never deleted, the embedding files are removed as
soon as a newer version of the checkpoint is fully committed.
