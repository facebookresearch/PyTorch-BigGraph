.. _dynamic-relations:

Dynamic relations
-----------------

.. caution:: This is an advanced topic!

Enabling the ``dynamic_relations`` flag in the configuration activates an alternative mode to be
used for graphs with a large number of relations (more than ~100 relations). In dynamic relation mode,
PBG runs with several modifications to its "standard" operation in order to support the large number of relations.
The differences are:

- The *number* of relations isn't provided in the config but is instead found in the input data, namely in the entity
  path, inside a :file:`dynamic_rel_count.txt` file. The settings of the relations, however, are still provided in the
  config file. This happens by providing a single relation config which will act as a "template" for all other ones, by
  being duplicated an appropriate number of times. One can think of this as the one relation in the config being
  "broadcasted" to the size of the relation list found in the :file:`dynamic_rel_count.txt` file.

- The batches of positive edges that are passed from the training loop into the model contain edges for multiple relation
  types at the same time (instead of each batch coming entirely from the same relation type). This introduces some performance challenges
  in how the operators are applied to the embeddings, as instead of a single operator with a single set of parameters
  applied to all edges, there might be a different one for each edge. The previous property ensures that all the operators
  are of the same type, so just their parameters might differ from one row to another. To account for this, the operators
  for dynamic relations are implemented differently, with a single operator object containing the parameters for all
  relation types. This implementation detail should be transparent as for how the operators are applied to the embeddings,
  but might come up when retrieving the parameters at the end of training.

- With non-dynamic relations, the operator is applied to the embedding of the right-hand side entity of the edge, whereas
  the embedding of the left-hand side entity is left unchanged. In a given batch, denote the :math:`i`-th positive edge
  by :math:`(x_i, r, y_i)` (:math:`x_i` and :math:`y_i` being the left- and right-hand side entities, :math:`r` being the
  relation type). For each of the positive edges, denote its :math:`j`-th negative sample :math:`(x_i, r, y'_{i,j})`.
  Due to :ref:`same-batch negative sampling <same-batch-negatives-sampling>` it may occur that the same right-hand side
  entity is used as a negative for several positives, that is, that :math:`y'_{i_1,j_1} = y'_{i_2,j_2}` for
  :math:`i_1 \neq i_2`. However, since it's the same relation type :math:`r` for all negatives, all the right-hand side
  entities will be transformed in the same way (i.e., passed through :math:`r`'s operator) no matter what positive edge
  they are a negative for. we need to apply the operator of :math:`r` to all of them, hence the total number of operator
  evaluations is equal to the number of positives and negatives.

  In case of dynamic relations the batch contains edges of the form :math:`(x_i, r_i, y_i)`, with possibly a different
  :math:`r_i` for each :math:`i`. If negative sampling and operator application worked the same, it might end up being
  necessary to transform each right-hand side entity multiple times in several ways, once for each different relation
  type of the edges the entity is a negative for. This would multiply the number of required operations by a significant
  factor and cause a sensible performance hit.

  To counter this, operators are applied differently in case of dynamic relations. They are applied to *either* the
  left- *or* the right-hand side (never both at the same time), and a different set of parameters is used in each of
  these two cases. On an input edge :math:`(x_i, r_i, y_i)` both ways of applying the operators are performed (separately).
  For the negatives of the form :math:`(x'_{i,j}, r_i, y_i)` (i.e., with the left-hand side entity changed), the operator
  is only applied to the right-hand side. Symmetrically, on :math:`(x_i, r_i, y'_{i,j})`, the operator is only applied to
  the left-hand side. This means that the operator is ever only applied to the entities of the original positive input
  edge, not on the entities of the negatives. Thus the number of operator evaluations is equal to the number of input
  edges in the batch.

  One could imagine it as if, for each edge of a certain relation type, a reversed edge were added to the graph, of a
  symmetric relation type. For each of these edges, the operator is only applied to the right-hand side, just like with
  standard relations. However, when sampling negatives, only the left-hand side entities are replaced, whereas the
  right-hand ones are kept unchanged.

  For more insight about this, look also at the "reciprocal predicates" described in [this paper](https://arxiv.org/pdf/1806.07297.pdf).
