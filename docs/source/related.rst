Related Works
=============

PBG is built on top of several Knowledge Base Completion papers, integrating what
has been shown to work well over time. In the sections below, we describe several
models that inspired some of the operators and options of PBG.

TransE
______

TransE_ is a popular model in Knowledge Base Completion due to its simplicity:
embeddings and relations are represented as vectors :math:`\theta_x` in
:math:`\mathbb{R}^D` and the score of a triple :math:`(x, r, y)` is computed as :

.. math::
    s(x, r, y) = d(\theta_x + \theta_r - \theta_y)

where :math:`d` is a dissimilarity function such as the :math:`L_1` or :math:`L_2`
norm. This model corresponds to the ``translation`` operator in PBG.

PBG only provides ``dot`` and ``cos`` as comparators, but dissimilarity functions
can easily be introduced as explained in :ref:`comparators <comparators>`.

RESCAL
______

RESCAL_ is a restriction on the Tucker factorization. Relations are represented
as matrices :math:`M_r`, and the score of a triple :math:`(x, r, y)` is computed as :

.. math::
    s(x, r, y) = \theta^{\top}_x M_r \theta_y

This corresponds to the ``linear`` operator. The original paper suggests to use
weight-decay on the parameters of the model. Such regularization is not available
in PBG currently, which relies instead on early stopping and control of the maximum
norm of the embeddings which scales more easily.


DistMult
________
DistMult_ is a special case of RESCAL, in which relations are limited to diagonal
operators represented as vectors :math:`\theta_r`. The score of a triple :math:`(x, r, y)`
is computed as :

.. math::
    s(x, r, y) = \langle \theta_x, \theta_r, \theta_y\rangle = \sum_{d=1}^D \theta_{x, d}\theta_{r, d}\theta_{y, d}

This is the ``diagonal`` operator in PBG. Notice that with similar
embeddings on the left and right-hand side, this operator is limited to representing
symmetric relations. This restrictions however leads to less over-fitting and good
performances on `several benchmarks <https://arxiv.org/abs/1705.10744>`_.


ComplEx
_______

ComplEx_ is similar to DistMult, but uses embeddings in :math:`\mathbb{C}` and
represents right-hand side embeddings as complex conjugates of left-hand side ones.
This allows representation of non-symmetric relations. The score of a triple :math:`(x, r, y)`
is computed as :

.. math::
    s(x, r, y) = Re(\langle \theta_x, \theta_r, \overline{\theta_y}\rangle)

The ``complex_diagonal`` operator in PBG interprets embeddings as :math:`D/2`
dimensional vectors, with the first :math:`D/2` values representing the real part,
and the remaining values the imaginary part. As shown in the original paper, the
ComplEx score can then be written as a dot product in :math:`\mathbb{R}^{D}`.

Reciprocal Relations
____________________
Two papers (`[1] <https://arxiv.org/abs/1806.07297>`_ and `[2] <https://arxiv.org/abs/1802.04868>`_)
simultaneously suggested to explicitly train on reciprocal relations. This can
be done in PBG with :ref:`dynamic relations <dynamic-relations>`. Jointly with the ``complex_diagonal``
operator, this allows reproducing state of the art results on ``FB15K`` with PBG.

.. _TransE: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
.. _RESCAL: http://www.icml-2011.org/papers/438_icmlpaper.pdf
.. _Distmult: https://arxiv.org/pdf/1412.6575v4.pdf
.. _ComplEx: http://proceedings.mlr.press/v48/trouillon16.pdf