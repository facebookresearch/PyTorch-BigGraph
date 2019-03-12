Evaluation
==========

During training, the average loss is reported for each edge bucket at each pass.
Evaluation metrics can be computed on held-out data during or after training to
measure the quality of trained embeddings.

Offline evaluation
------------------

The ``torchbiggraph_eval`` command will perform an offline evaluation of trained PBG embeddings on a validation dataset.
This dataset should contain held-out data not included in the training dataset. It is invoked in the same
way as the training command and takes the same arguments.

It is generally advisable to have two versions of the config file, one for training and one for evaluation, with the same
parameters except for the edge paths, in order to evaluate a separate (and often smaller) set of edges. (It's also possible
to use a single config file and have it produce different output based on environment variables or other context).
Training-specific config parameters (e.g., the learning rate, loss function, ...) will be ignored during evaluation.

The metrics are first reported on each bucket, and a global average is computed at the end.
(If multiple edge paths are in use, metrics are computed separately for each of them but still ultimately averaged).

Many metrics are statistics based on the "ranks" of the edges of the validation set.
The rank of a positive edge is determined by the rank of its score against the scores of
:ref:`a certain number of negative edges <negative-sampling>`. A rank of 1 is the "best"
outcome as it means that the positive edge had a higher score than all the negatives. Higher
values are "worse" as they indicate that the positive didn't stand out.

It may happen that some of the negative samples used in the rank computation are in fact
other positive samples, which are expected to have a high score and may thus cause adverse effects on the rank.
This effect is especially visible on smaller graphs, in particular when all other entities are used to construct the negatives.
To fix it, and to match what is typically done in the literature,
a so-called "filtered" rank is used in the FB15k demo script (and there only), where positive
samples are filtered out when computing the rank of an edge. It is hard to scale this technique
to large graphs, and thus it is not enabled globally. However, filtering is less important
on large graphs as it's less likely to see a training edge among the sampled negatives.

The metrics are:

- **Mean Rank**: the average of the ranks of all positives (lower is better, best is 1).
- **Mean Reciprocal Rank (MRR)**: the average of the *reciprocal* of the ranks of all positives (higher is better, best is 1).
- **Hits@1**: the fraction of positives that rank better than all their negatives, i.e., have a rank of 1 (higher is better, best is 1).
- **Hits@10**: the fraction of positives that rank in the top 10 among their negatives (higher is better, best is 1).
- **Hits@50**: the fraction of positives that rank in the top 50 among their negatives (higher is better, best is 1).
- **Area Under the Curve (AUC)**: an estimation of the probability that a randomly chosen positive scores higher than a
  randomly chosen negative (*any* negative, not only the negatives constructed by corrupting that positive).

.. _evaluation-during-training:

Evaluation during training
--------------------------

Offline evaluation is a slow process that is intended to be run after training is complete
to evaluate the final model on a held-out set of edges constructed by the user. However, it's
useful to be able to monitor overfitting as training progresses. PBG offers this functionality,
by calculating the same metrics as the offline evaluation before and after each pass on a
small set of training edges. These stats are printed to the logs.

The metrics are computed on a set of edges that is held out automatically from the training set. To be more explicit:
using this feature means that training happens on *fewer* edges, as some are excluded and reserved for this evaluation.
The holdout fraction is controlled by the ``eval_fraction`` config parameter (setting it to zero thus disables this
feature). The evaluations before and after each training iteration happen on the same set of edges, thus are comparable.
Moreover, the evaluations for the same edge chunk, edge path and bucket at different epochs also use the same set of edges.

Evaluation metrics are computed both before and after training each edge bucket because it provides insight into
whether the partitioned training is working. If the partitioned training is converging, then the gap between the "before"
and "after" statistics should go to zero over time. On the other hand, if the partitioned training is causing the model to
overfit on each edge bucket (thus decreasing performance for other edge buckets) then there will be a persistent gap between
the "before" and "after" statistics.

It's possible to use different batch sizes for :ref:`same-batch <same-batch-negatives-sampling>` and
:ref:`uniform negative sampling <same-batch-negatives-sampling>` by tuning the ``eval_num_batch_negs`` and the
``eval_num_uniform_negs`` config parameters.
