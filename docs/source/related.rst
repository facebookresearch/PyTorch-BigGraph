Related works
=============

PBG was designed with the expertise gained from many previous works in the knowledge
base completion literature, integrating what has been shown to work well over time.
In the sections below, we describe the models that inspired some of the operators and features of PBG.

TransE
------

TransE_ is a popular model in knowledge base completion due to its simplicity:
when two embeddings are compared to calculate the score of an edge between them,
the right-hand side one is first translated by a vector :math:`v_r` (of the same
dimension as the embeddings) that is specific to the relation type. Contrary to
PBG, TransE aims at giving lower scores to entities that are nearby, hence the
score of a triple :math:`(x, r, y)` is computed as:

.. math::
    s(x, r, y) = d(\theta_x + v_r - \theta_y)

where :math:`d` is a dissimilarity function such as the :math:`L_1` or :math:`L_2`
norm.

PBG can be configured to operate like TransE by using the ``translation`` operator
and by introducing a new :ref:`comparator <comparators>` based on the desired
dissimilarity function. However, contrary to the dot product or the cosine distance,
the comparison between all pairs of vectors from two sets using the :math:`L_1` and
:math:`L_2` norms cannot be expressed as a matrix multiplication and thus would be
challenging to implement as efficiently. One could consider using the cosine distance
instead of the :math:`L_2` norm since the former measures (the cosine of) the angle
between two vectors which, when small, is approximately their :math:`L_2` distance.

RESCAL
------

RESCAL_ is a restriction on the Tucker factorization. Relations are represented
as matrices :math:`M_r`, and the score of a triple :math:`(x, r, y)` is computed as:

.. math::
    s(x, r, y) = \theta^{\top}_x M_r \theta_y

This corresponds to PBG's ``linear`` operator. The original paper suggests to use
weight-decay on the parameters of the model. Such regularization is not available
in PBG currently, which relies instead on early stopping and control of the maximum
norm of the embeddings which scales more easily.

The need for weight decay stems from each relation having a lot of parameters,
which could lead them to overfit and to not perform well on, for example, FB15k.
RESCAL should only be considered for models with a large number of edges for
each relation type, where overfitting is not an issue.

DistMult
--------

DistMult_ is a special case of RESCAL, in which relations are limited to diagonal
matrices represented as vectors :math:`v_r`. The score of a triple :math:`(x, r, y)`
is thus:

.. math::
    s(x, r, y) = \langle \theta_x, v_r, \theta_y \rangle = \sum_{d=1}^D \theta_{x, d} v_{r, d} \theta_{y, d}

This is the ``diagonal`` operator in PBG. Notice that with the same embedding
space on the left and right-hand side, this operator is limited to representing
symmetric relations. This restriction however leads to less over-fitting and good
performances on `several benchmarks <https://arxiv.org/abs/1705.10744>`_.

ComplEx
-------

ComplEx_ is similar to DistMult, but uses embeddings in :math:`\mathbb{C}` and
represents the right-hand side embeddings as complex conjugates of the left-hand side ones.
This allows to represent non-symmetric relations. The score of a triple :math:`(x, r, y)`
is computed as:

.. math::
    s(x, r, y) = \operatorname{Re}(\langle \theta_x, v_r, \overline{\theta_y} \rangle)

The ``complex_diagonal`` operator in PBG interprets a :math:`D`-dimensional real
embedding as a :math:`D/2`-dimensional complex one, with the first :math:`D/2`
values representing the real part and the remaining ones for the imaginary part.
As shown in the original paper, the ComplEx score can then be written as a dot
product in :math:`\mathbb{R}^{D}`, hence replicated in PBG using the ``dot`` operator.

Reciprocal Relations
--------------------

Two papers (`[1] <https://arxiv.org/abs/1806.07297>`_ and `[2] <https://arxiv.org/abs/1802.04868>`_)
simultaneously suggested to explicitly train on reciprocal relations, i.e., for each triple
:math:`(x, r, y)` in the training set add another one :math:`(y, r', x)`. This can
be done implicitly in PBG with :ref:`dynamic relations <dynamic-relations>`. Jointly with
the ``complex_diagonal`` operator, this allows reproducing state of the art results on ``FB15K`` with PBG.

.. _TransE: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
.. _RESCAL: http://www.icml-2011.org/papers/438_icmlpaper.pdf
.. _Distmult: https://arxiv.org/pdf/1412.6575v4.pdf
.. _ComplEx: http://proceedings.mlr.press/v48/trouillon16.pdf
