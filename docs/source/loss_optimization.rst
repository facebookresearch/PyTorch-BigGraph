.. _loss-calculation:

Loss calculation
================

The training process aims at finding the embeddings for the entities so that the scores of the positive edges are higher
than the scores of the negative edges. When unpacking what this means, three different aspects come into play:

- One must first determine which edges are to be considered as **positive and negative samples**.
- Then, once the scores of all the samples have been determined, one must decide how to aggregate them in a single **loss** value.
- Finally, one must decide how to go about **optimizing** that loss.

This chapter will dive into each of these issues.

.. _negative-sampling:

Negative sampling
-----------------

The edges provided in the input data are known to be positives but, as PBG operates under the open-world assumption, the
edges that are not in the input are not necessarily negatives. However, as PBG is designed to perform on large sparse
graphs, it relies on the approximation that any random edge is a negative with very high probability.

The goal of sampling negatives is to produce a set of negative edges for each positive edge of a batch. Usual downstream
applications (ranking, classification, ...) are interested in comparing the score of an edge :math:`(x, r, y_1)` with the
score of an edge :math:`(x, r, y_2)`. Therefore, PBG produces negative samples for a given positive edge by corrupting the
entity on one of its sides, keeping the other side and the relation type intact. This makes the sampling more suited to the task.

For performance reasons, the set of entities used to corrupt the positive edges in order to produce the negative samples
may be shared across several positive edges. The way this usually happens is that positive edges are split into "chunks",
a single set of entities is sampled for each chunk, and all edges in that chunk are corrupted using that set of entities.

PBG supports several ways of sampling negatives:

.. _all-negatives-sampling:

All negatives
^^^^^^^^^^^^^

The most straightforward way is to use, for each positive edge, *all* its possible negatives. What this means is that
for a positive :math:`(x, r, y)` (where :math:`x` and :math:`y` are the left- and right-hand side negatives respectively
and :math:`r` is the relation type), its negatives will be :math:`(x', r, y)` for all :math:`x'` of the same entity type
as :math:`x` and :math:`(x, r, y')` for :math:`y'` of the same entity type as :math:`y`. (Due to technical reasons this
is in fact restricted to only the :math:`x'` in the same partition as :math:`x`, and similarly for :math:`y'`, as
negative sampling always operates within the current bucket.)

As one can imagine, this method generates a lot of negatives and thus doesn't scale to graphs of any significant size.
It should not be used in practice, and is provided in PBG mainly for "academic" reasons. It is mainly useful to get
more accurate results during evaluation on small graphs.

This method is activated on a per-relation basis, by turning on the ``all_negs`` config flag. When it's enabled, this
mode takes precedence and overrides any other mode.

.. _same-batch-negatives-sampling:

Same-batch negatives
^^^^^^^^^^^^^^^^^^^^

This negative sampling method produces negatives for a given positive edge of a batch by sampling from the other edges
of the same batch. This is done by first splitting the batch into so-called chunks (beware that the name "chunks" is
overloaded, and these chunks are different than the edge chunks explained in :ref:`batch-preparation`). Then the set of
negatives for a positive edge :math:`(x, r, y)` contains the edges :math:`(x', r, y)` for all entities :math:`x'` that
are on left-hand side of another edge in the chunk, and the edges :math:`(x, r, y')` with :math:`y'` satisfying the same
condition for the right-hand side.

For a single positive edge, this means that the entities used to construct its negatives are sampled from the current
partition proportionally to their degree, a.k.a., according to the data distribution. This helps counteract the effects
of a very skewed distribution of degrees, which might cause the embeddings to just capture that distribution.

The size of the chunks is controlled by the global ``num_batch_negs`` parameter. To disable this sampling mode, set that
parameter to zero.

.. _uniform-negatives-sampling:

Uniformly-sampled negatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This last method is perhaps the most natural approximation of the "all negatives" method that scales to arbitrarily large
graphs. Instead of using *all* the entities on either side to produce negative edges (thus having the number of negatives
scale linearly withe the size of the graph), a fixed given number of these entities is sampled uniformly with replacement.
Thus the set of negatives remains of constant size no matter how large the graph is. As with the "all negatives" method,
the sampling here is restricted to the entities that have the same type and that belong to the same partition as the
entity of the positive edge.

This method interacts with the same-batch method, as all the edges in a chunk receive the same set of uniformly sampled
negatives. This caveat means that the uniform negatives of two different positives are independent and uncorrelated only
if they belong to different chunks.

This method is controlled by the ``num_uniform_negs`` parameter, which controls how many negatives are sampled for each
chunk. If ``num_batch_negs`` is zero, the batches will be split into chunks of size ``num_uniform_negs``.

.. _loss:

Loss functions
--------------

Once positive and negative samples have been determined and their scores have been computed by the model, the scores'
suitability for a certain application must be assessed, which is done by aggregating them into a single real value, the
loss. What loss function is most appropriate depends on what operations the embeddings will be used for.

In all cases, the loss function's input will be a series of scores for positive samples and, for each of them, a set of
scores for corresponding negative samples. For simplicity, suppose all these sets are of the same size (if they are not,
they can be padded with "negative infinity" values, as these are the "ideal" scores for negative edges and should thus
induce no loss).

Ranking loss
^^^^^^^^^^^^

The ranking loss compares each positive score with each of its corresponding negatives. For each such comparison, it
calculates how much the positive score is above the negative one. If this distance is larger than a given target margin,
no loss is introduced. Otherwise, the loss caused by this pair of scores is the amount by which this margin is violated.
Formally, for a margin :math:`m`, a positive score :math:`s_i` and a negative score :math:`t_{i,j}`, the loss is
:math:`\max(0, m - s_i + t_{i,j})`. The total loss is the sum of the losses over all pairs of positive and negative
scores, i.e., over all :math:`i` and :math:`j`.

This loss function is chosen by setting the ``loss_fn`` parameter to ``ranking``, and the target margin is specified by
the ``margin`` parameter.

This loss function is suitable when the setting requires to rank some entities by how likely they are to be related to
another given entity.

Logistic loss
^^^^^^^^^^^^^

The logistic loss instead interprets the scores as the probabilities that the edges exist. It does so by first passing
each score (whose domain is the entire real line) through the logistic function (:math:`x \mapsto 1 / (1 + e^{-x})`,
which maps it to a value between 0 and 1). This value is taken as the probability :math:`p` and the loss will be its
binary cross entropy with the "target" probability, i.e., 1 for positive edges and 0 for negative ones. In formulas, the
loss for positives is :math:`- \log p` whereas for negatives it's :math:`- \log (1 - p)`.

One can see this as the cross entropy between two distributions on the values "edge exists" and "edge doesn't exist".
One is given by the score (passed through the logistic function), the other has all the mass on "exists" for positives
or all the mass on "doesn't exist" for negatives.

This loss function is parameterless and is enabled by setting ``loss_fn`` to ``logistic``.

Softmax loss
^^^^^^^^^^^^

The last loss function is designed for when one wants a distribution on the probabilities of some entities being related
to a given entity (contrary to just wanting a ranking, as with the ranking loss). For a certain positive :math:`i`, its
score :math:`s_i` and the score :math:`t_{i,j}` of all the corresponding negatives :math:`j` are first converted to
probabilities by performing a softmax: :math:`p_i \propto e^{s_i}` and :math:`q_{i,j} \propto e^{t_{i,j}}`, normalized
so that they sum up to 1. Then the loss is the cross entropy between this distribution and the "target" one, i.e., the
one that puts all the mass on the positive sample. So, in full, the loss for a single :math:`i` is :math:`- \log p_i`,
i.e., :math:`- s_i + \log \sum_j e^{t_{i,j}}`.

This loss is activated by setting ``loss_fn`` to ``softmax``.

.. _optimizers:

Optimizers
----------

The `Adagrad <http://jmlr.org/papers/v12/duchi11a.html>`_ optimization method is used to update all model parameters. Adagrad performs stochastic gradient descent with an adaptive learning rate applied to each parameter inversely proportional to the inverse square magnitude of all previous updates. In practice, Adagrad updates lead to an order of magnitude faster convergence for typical PBG models.

The initial learning rate for Adagrad is specified by the `lr` config parameter.A separate learning rate can also be set for non-embeddings using the `relation_lr` parameter.

Standard Adagrad requires an equal amount of memory for optimizer state as the size of the model, which is prohibitive for the large models targeted by PBG. To reduce optimizer memory usage, a modified version of Adagrad is used that uses a common learning rate for each entity embedding. The learning rate is proportional to the inverse sum of the squared gradients from each element of the embedding, divided by the dimension. Non-embedding parameters (e.g. relation operator parameters) use standard Adagrad.

Adagrad parameters are updated asynchronously across worker threads with no explicit synchronization. Asynchronous updates to the Adagrad state (the total squared gradient) appear stable, likely because each element of the state tensor only accumulates positives updates. Optimization is further stabilized by performing a short period of training with a single thread before beginning Hogwild! training, which is tuned by the ``hogwild_delay`` parameter.

In distributed training, the Adagrad state for shared parameters (e.g. relation operator parameters) are shared via the parameter server using the same asynchronous gradient update as the parameters themselves. Similar to inter-thread synchronization, these asynchronous updates are stable after an initial burn-in period because the total squared gradient strictly accumulates positive values.
