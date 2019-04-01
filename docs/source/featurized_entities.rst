.. _featurized-entities:

Featurized entities
===================

.. caution:: This is an advanced feature, which is still under development and hasn't fully stabilized yet.

In normal operation PBG considers each entity atomic and distinct from all others, and as such it learns an independent embedding
for each of them, with no correlation other than the one acquired during training. However, it is common practice to represent
some type of data as collections of underlying "features", each of which has its own learned embedding. The embedding of an entity
will be implicitly derived from the embeddings of its features. Sharing a feature will enforce a correlation between the embeddings
of two entities.

For example, entities that represent text documents could have their words as features, i.e., an embedding is learned
for each word and the embedding of a document is the average of the embeddings of the words it contains.

PBG provides this capability. Featurized mode is activated on a per-entity type basis by enabling the
``featurized`` flag on its config. As this feature isn't finalized yet, the tooling around it isn't up to par
with non-featurized entities, in particular for converting featurized edgelists to the PBG format.
Practitioners will have to implement their own converters, based on the format described below.
Contributions of converters to and from standard formats are welcome.

The following changes occur in the training process when featurized entities are enabled:

- The count stored in the :file:`entity_count_{type}_{part}.txt` file refers to the total number of different *features*
  that are encountered in the edge files, rather than to the number of different sets of features.

- Each edge file :file:`edges_{lhs}_{rhs}.h5` must contain a few more datasets. If any edge in it has a featurized
  entity on the left-hand side then it must contain two one-dimensional datasets of integers: ``lhsd_data``, which
  contains the flattened concatenation of the lists of features of all left-hand side entities of the edges in the file,
  and ``lhsd_offsets``, which contains the "cutpoints" of ``lhsd_data`` where the feature list of one entity ends and
  the one for the next entity starts.

  Thus the *entries* of ``lhsd_data`` are feature identifiers, while the *entries* of ``lhsd_offsets`` are *indices* of
  ``lhsd_data``. Each pair of consecutive entries of ``lhsd_offsets`` represents an half-open interval of ``lhsd_data``,
  thus the first entry of ``lhsd_offsets`` should be 0, the last entry should be the size of ``lhsd_data``, and entries
  should be in non-decreasing order. If the edge file contains :math:`N` edges, then ``lhsd_offsets`` must contain
  :math:`N + 1` entries.

  * If the left-hand side entity of edge :math:`i` is featurized, then its features will be the values of ``lhsd_data``
    between positions ``lhsd_offsets``:math:`[i]` (inclusive) and ``lhsd_offsets``:math:`[i+1]` (exclusive).
    The :math:`i`-th entry of the ``lhs`` dataset, on the other hand, can be any value, as it will be ignored.

  * If the left-hand side entity of edge :math:`i` is *not* featurized, then the offset of the entity will be in
    ``lhs``:math:`[i]`, just as usual. In that case its set of features should be empty, that is, one should have
    ``lhsd_offsets``:math:`[i]` equal to ``lhsd_offsets``:math:`[i+1]`.

  If any right-hand side entity is featurized, the same must hold for datasets ``rhsd_offsets`` and ``rhsd_data``.

- Entities are represented as "bags of features". That is, their embeddings will be the average of the embeddings of their
  features, similarly to how text documents can be represented as the average of the embeddings of the words they contain.

- The only form of :ref:`negative sampling <negative-sampling>` supported for featurized entities is the
  :ref:`same-batch mode <same-batch-negatives-sampling>`. Both the :ref:`all negatives <all-negatives-sampling>` and the
  :ref:`uniformly-sampled negatives mode <uniform-negatives-sampling>` are not supported. Observe that this means that
  uniform sampling of negatives must be disabled globally.
