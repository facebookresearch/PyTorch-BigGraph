Downstream tasks
================

PBG is intended to be a tool that takes as input a graph and produces as output embeddings for it.
It should not be PBG's concern to know where the graph comes from or what it represents, or to deal
with how the embeddings are going to be utilized for the task at hand. (One exception is the loss
function, which should be chosen based on the intended use of the embeddings).

In a sense, PBG should be seen as a block, that must be chained with other blocks to form a full
pipeline. As such, PBG aims to be self-contained, to do one job (and do it well) and to adhere to
standard interfaces so that it can more easily be plugged to other utilities.

PBG does not intend to add support for "downstream tasks" (querying, ranking, clustering, nearest
neighbor search, ...) to the main codebase, as there are plenty of them and everyone wants them
in a slightly different flavor. Trying to properly support them all is impossible and would bloat
the code and dilute the core functionality.

However, in order to provide a starting point for users to write their own downstream tools, this
page gives some basic examples of how to consume PBG's output data and perform some toy tasks on it.

Parsing the output data
-----------------------

PBG's input and output is entirely file-based, thus its "interface" specification consists in these
files' formats. Its "native" format is quite custom but is based on the standard HDF5 format. The
specifications are in the :ref:`io-format` section, and the format is designed to keep backwards
compatibility. To ease interoperability, PBG tries to provide converters to and from other common formats.
For now, the only such converter is for a text-based TSV format (tab-separated values). Contributions
for additional converters are welcome.

PBG expects its input to be already partitioned, and identifies each entity solely by its type, the
partition it belongs to and its offset within that partition. Most graphs don't naturally come in this
form: entities may, for example, be represented just by a name (a string). Third-party converters need
to be written or employed to transform a graph from its native representation to PBG's one, and to then
"match" the output embeddings back with their original entities. The TSV converters do this and can be
used as an example.

Reading the HDF5 format
^^^^^^^^^^^^^^^^^^^^^^^

Suppose that you have completed the training of the :file:`examples/fb15k.py` script and want to now
look up the embedding of some entity. For that, we'll need to read:

- the embeddings, from the checkpoint files (the :file:`.h5` files in the `model/fb15k` directory, or
  whatever directory was specified as the ``checkpoint_path``); and
- the mapping from entity names to their partitions and offsets, from the :file:`data/FB15k/dictionary.json`
  file created by the ``torchbiggraph_import_from_tsv`` command.

The embedding of, say, entity ``/m/05hf_5`` can be found as follows::

    import json
    import h5py

    with open("data/FB15k/dictionary.json", "rt") as tf:
        dictionary = json.load(tf)
    offset = dictionary["entities"]["all"].index("/m/05hf_5")

    with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
        embedding = hf["embeddings"][offset, :]

    print(embedding)

The HDF5 format allows partial reads so, in the example above, we only load from disk the data we actually
need. If we wanted to load the entire embedding table we could use ``embeddings = hf["embeddings"][...]``.

Reading from the TSV format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose that, instead, you just have the TSV format produced by ``torchbiggraph_export_to_tsv``. For example
because you downloaded the :ref:`pre-trained Wikidata embeddings <wiki-data>` (available
`here <https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz>`_ gzipped).
This format is textual (i.e., ASCII-encoded). Each line contains first a string and then a series of real
numbers (in floating-point decimal form), all separated by tabs.

- First come the entities' embeddings, with the string being the entity name and the rest being the
  entity's embedding.
- Then come the parameters of the relation types' operators, with the string being the relation type name
  and the rest being a flattened view of the parameters.
- Then, if :ref:`dynamic relations <dynamic-relations>` are used, come the parameters for the automatically-generated
  reversed relation types.

In the Wikidata embeddings (and _not_ in other TSV files) the first line contains a comment listing the entity count,
the relation count and the dimension.

If one wants to load the data of such a file from disk into a NumPy array (just the embeddings, without the labels)
one can use the following command::

    import numpy as np

    embeddings = np.loadtxt(
        "wikidata_translation_v1.tsv",
        dtype=np.float32,
        delimiter="\t",
        skiprows=1,
        max_rows=78404883,
        usecols=range(1, 201),
        comments=None,
    )

Let's break it down:

- ``delimiter`` specifies what character to use to split a single line into fields. As these are
  tab-separated values, the character must be a tab.
- ``skiprows`` makes NumPy ignore the first row, because for the Wikidata embeddings it contains
  a comment. In other cases one should omit ``skiprows`` or set it to zero.
- ``max_rows`` causes NumPy to load only the first 78404883 rows (after skipping the first one).
  That number isn't magic, it's simply the number of entities in the Wikidata dataset, and we need
  it in order to load all and only the entity embeddings, without loading the relation type parameters.
- ``usecols`` tells NumPy to ignore the first column, which contains the entity name, and instead
  use the next 200 columns. We use 200 because that's the dimension of the Wikidata embeddings.
- ``comments`` by default is ``#`` and NumPy will ignore everything that comes after the first
  occurrence of that character, however some Wikidata entities contain ``#`` in their names thus we
  must unset this value to have NumPy properly parse the row.

Be warned however that parsing such a text file is a very slow operation. In fact, the TSV format is
mainly helpful for small datasets, and is intended for demonstrative purposes, not for actual usage
in a performance-sensitive scenario.

Reading from the NPY format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, for example in the :ref:`Wikidata embeddings <wiki-data>`, we also provide a :file:`.npy`
file containing the embeddings. This data is the same that would be obtained by the ``loadtxt`` function
above, except that the hard work of parsing has already been done and the format is now easily machine-readable
and thus more performant. It can be loaded easily as follows::

    import numpy as np

    embeddings = np.load("wikidata_translation_v1_vectors.npy")

This loads all the data in memory. If one only wants to access some part of the data, one can play with the
``mmap_mode`` option so that the data remains on disk until actually accessed.

Using the embeddings
--------------------

Predicting the score of an edge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in the :ref:`scoring` section, the essential goal of the model at the code of PBG is to be able
to assign a score to each triplet of source entity, target entity and relation type. Those scores should reflect
the likelihood of that edge existing. PBG's current code for calculating these scores is very intertwined with
the code that samples negative edges and therefore it is hard to use a trained model just to predict scores.

The following code shows loads the data directly from the HDF5 files and manually calculate the score of Paris
being the capital of France::

    import json
    import h5py
    import torch
    from torchbiggraph.model import ComplexDiagonalDynamicOperator, DotComparator

    # Load count of dynamic relations
    with open("data/FB15k/dynamic_rel_count.txt", "rt") as tf:
        dynamic_rel_count = int(tf.read().strip())

    # Load the operator's state dict
    with h5py.File("model/fb15k/model.v50.h5", "r") as hf:
        operator_state_dict = {
            "real": torch.from_numpy(hf["model/relations/0/operator/rhs/real"][...]),
            "imag": torch.from_numpy(hf["model/relations/0/operator/rhs/imag"][...]),
        }
    operator = ComplexDiagonalDynamicOperator(400, dynamic_rel_count)
    operator.load_state_dict(operator_state_dict)
    comparator = DotComparator()

    # Load the offsets of the entities and the index of the relation type
    with open("data/FB15k/dictionary.json", "rt") as tf:
        dictionary = json.load(tf)
    src_entity_offset = dictionary["entities"]["all"].index("/m/0f8l9c")  # France
    dest_entity_offset = dictionary["entities"]["all"].index("/m/05qtj")  # Paris
    rel_type_index = dictionary["relations"].index("/location/country/capital")

    # Load the trained embeddings
    with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
        src_embedding = torch.from_numpy(hf["embeddings"][src_entity_offset, :])
        dest_embedding = torch.from_numpy(hf["embeddings"][dest_entity_offset, :])

    # Calculate the scores
    scores, _, _ = comparator(
        comparator.prepare(src_embedding.view(1, 1, 400)),
        comparator.prepare(
            operator(
                dest_embedding.view(1, 400),
                torch.tensor([rel_type_index]),
            ).view(1, 1, 400),
        ),
        torch.empty(1, 0, 400),  # Left-hand side negatives, not needed
        torch.empty(1, 0, 400),  # Right-hand side negatives, not needed
    )

    print(scores)

Ranking
^^^^^^^

A very related problem is, given a source entity and a relation type, ranking all the entities by how likely they are
to be the target entity. This can be done very similarly to the above. For example, the following code determines what
entities are most likely to be the capital of France::

    import json
    import h5py
    import torch
    from torchbiggraph.model import ComplexDiagonalDynamicOperator, DotComparator

    # Load entity count
    with open("data/FB15k/entity_count_all_0.txt", "rt") as tf:
        entity_count = int(tf.read().strip())

    # Load count of dynamic relations
    with open("data/FB15k/dynamic_rel_count.txt", "rt") as tf:
        dynamic_rel_count = int(tf.read().strip())

    # Load the operator's state dict
    with h5py.File("model/fb15k/model.v50.h5", "r") as hf:
        operator_state_dict = {
            "real": torch.from_numpy(hf["model/relations/0/operator/rhs/real"][...]),
            "imag": torch.from_numpy(hf["model/relations/0/operator/rhs/imag"][...]),
        }
    operator = ComplexDiagonalDynamicOperator(400, dynamic_rel_count)
    operator.load_state_dict(operator_state_dict)
    comparator = DotComparator()

    # Load the offsets of the entities and the index of the relation type
    with open("data/FB15k/dictionary.json", "rt") as tf:
        dictionary = json.load(tf)
    src_entity_offset = dictionary["entities"]["all"].index("/m/0f8l9c")  # France
    rel_type_index = dictionary["relations"].index("/location/country/capital")

    # Load the trained embeddings
    with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
        src_embedding = torch.from_numpy(hf["embeddings"][src_entity_offset, :])
        dest_embeddings = torch.from_numpy(hf["embeddings"][...])

    # Calculate the scores
    scores, _, _ = comparator(
        comparator.prepare(src_embedding.view(1, 1, 400)).expand(1, entity_count, 400),
        comparator.prepare(
            operator(
                dest_embeddings,
                torch.tensor([rel_type_index]).expand(entity_count),
            ).view(1, entity_count, 400),
        ),
        torch.empty(1, 0, 400),  # Left-hand side negatives, not needed
        torch.empty(1, 0, 400),  # Right-hand side negatives, not needed
    )

    # Sort the entities by their score
    permutation = scores.flatten().argsort(descending=True)
    top5_entities = [dictionary["entities"]["all"][index] for index in permutation[:5]]

    print(top5_entities)

Which in my case gives, in order, `Paris <https://www.wikidata.org/wiki/Q90>`_,
`Lyon <https://www.wikidata.org/wiki/Q456>`_, `Martinique <https://www.wikidata.org/wiki/Q17054>`_,
`Strasbourg <https://www.wikidata.org/wiki/Q6602>`_ and `Rouen <https://www.wikidata.org/wiki/Q30974>`_.

Nearest neighbor search
^^^^^^^^^^^^^^^^^^^^^^^

Another common task is finding the entities whose embeddings are the closest to a given target vector.
In order to do the actual search, we will use the `FAISS <https://github.com/facebookresearch/faiss>`_
library. The following code looks for the entities that are closest to Paris::

    import json
    import numpy as np
    import h5py
    import faiss

    # Create FAISS index
    index = faiss.IndexFlatL2(400)
    with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
        index.add(hf["embeddings"][...])

    # Get trained embedding of Paris
    with open("data/FB15k/dictionary.json", "rt") as f:
        dictionary = json.load(f)
    target_entity_offset = dictionary["entities"]["all"].index("/m/05qtj")  # Paris
    with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
        target_embedding = hf["embeddings"][target_entity_offset, :]

    # Search nearest neighbors
    _, neighbors = index.search(target_embedding.reshape((1, 400)), 5)

    # Map back to entity names
    top5_entities = [dictionary["entities"]["all"][index] for index in neighbors[0]]

    print(top5_entities)

Which in my case gives, in order, `Paris <https://www.wikidata.org/wiki/Q90>`_ (as expected),
`Louvre Museum <https://www.wikidata.org/wiki/Q19675>`_, `Helsinki <https://www.wikidata.org/wiki/Q1757>`_,
`Prague <https://www.wikidata.org/wiki/Q1085>`_ and `Montmartre Cemetery <https://www.wikidata.org/wiki/Q746647>`_.
