#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_extensions.tensorlist.tensorlist import TensorList

from .config import LossFunction, Operator, Comparator, EntitySchema, RelationSchema, ConfigSchema
from .util import log, Side


def match_shape(tensor, *expected_shape):
    """Compare the given tensor's shape with what you expect it to be.

    This function serves two goals: it can be used both to assert that the size
    of a tensor (or part of it) is what it should be, and to query for the size
    of the unknown dimensions. The former result can be achieved with:

        >>> match_shape(t, 2, 3, 4)

    which is similar to

        >>> assert t.size() == (2, 3, 4)

    except that it doesn't use an assert (and is thus not stripped when the code
    is optimized) and that it raises a TypeError (instead of an AssertionError)
    with an informative error message. It works with any number of positional
    arguments, including zero. If a dimension's size is not known beforehand
    pass a -1: no check will be performed and the size will be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, 2, -1, 4)
        3
        >>> match_shape(t, -1, 3, -1)
        (2, 4)

    If the number of dimensions isn't known beforehand, an ellipsis can be used
    as a placeholder for any number of dimensions (including zero). Their sizes
    won't be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, ..., 3, -1)
        4

    """
    if not all(isinstance(d, int) or d is Ellipsis for d in expected_shape):
        raise RuntimeError(
            "Some arguments aren't ints or ellipses: %s" % (expected_shape,))
    actual_shape = tensor.size()
    error = TypeError("Shape doesn't match: (%s) != (%s)" % (
        ", ".join("%d" % d for d in actual_shape),
        ", ".join("..." if d is Ellipsis else "*" if d < 0 else "%d" % d
                  for d in expected_shape)),
    )
    if Ellipsis not in expected_shape:
        if len(actual_shape) != len(expected_shape):
            raise error
    else:
        if expected_shape.count(Ellipsis) > 1:
            raise RuntimeError("Two or more ellipses in %s"
                               % (tuple(expected_shape),))
        if len(actual_shape) < len(expected_shape) - 1:
            raise error
        pos = expected_shape.index(Ellipsis)
        expected_shape = (expected_shape[:pos]
                          + actual_shape[pos:pos + 1 - len(expected_shape)]
                          + expected_shape[pos + 1:])
    unknown_dims: List[int] = []
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if expected_dim < 0:
            unknown_dims.append(actual_dim)
            continue
        if actual_dim != expected_dim:
            raise error
    if not unknown_dims:
        return None
    if len(unknown_dims) == 1:
        return unknown_dims[0]
    return tuple(unknown_dims)


class AbstractEmbedding(nn.Module, ABC):

    @abstractmethod
    def get_all_entities(self) -> torch.FloatTensor:
        pass

    @abstractmethod
    def sample_entities(self, *dims: int) -> torch.FloatTensor:
        pass


class SimpleEmbedding(AbstractEmbedding):

    def __init__(self, weight: nn.Parameter, max_norm: Optional[float]=None):
        super().__init__()
        self.weight: nn.Parameter = weight
        self.max_norm: Optional[float] = max_norm

    def forward(self, input: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(
            input, self.weight, max_norm=self.max_norm, sparse=True,
        )

    def get_all_entities(self) -> torch.FloatTensor:
        return self(torch.arange(self.weight.size(0)))

    def sample_entities(self, *dims: int) -> torch.FloatTensor:
        return self(torch.randint(low=0, high=self.weight.size(0), size=dims))


class FeaturizedEmbedding(AbstractEmbedding):

    def __init__(self, weight: nn.Parameter, max_norm: Optional[float]=None):
        super().__init__()
        self.weight: nn.Parameter = weight
        self.max_norm: Optional[float] = max_norm

    def forward(self, input: TensorList) -> torch.FloatTensor:
        if input.size(0) == 0:
            return torch.empty(0, self.weight.size(1))
        return F.embedding_bag(
            input.data.long(), self.weight, input.offsets[:-1],
            max_norm=self.max_norm, sparse=True,
        )

    def get_all_entities(self) -> torch.FloatTensor:
        raise NotImplementedError("Cannot list all entities for featurized entities")

    def sample_entities(self, *dims: int) -> torch.FloatTensor:
        raise NotImplementedError(
            "Cannot sample entities for featurized entities.")


class AbstractOperator(nn.Module, ABC):

    """Perform the same operation on many vectors.

    Given a tensor containing a set of vectors, perform the same operation on
    all of them, with a common set of parameters. The dimension of these vectors
    will be given at initialization (so that any parameter can be initialized).
    The input will be a tensor with at least one dimension. The last dimension
    will contain the vectors. The output is a tensor that will have the same
    size as the input.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        pass


class IdentityOperator(AbstractOperator):

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        return embeddings


class DiagonalOperator(AbstractOperator):

    def __init__(self, dim: int):
        super().__init__(dim)
        self.diagonal = nn.Parameter(torch.ones(self.dim))

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        return self.diagonal * embeddings


class TranslationOperator(AbstractOperator):

    def __init__(self, dim: int):
        super().__init__(dim)
        self.translation = nn.Parameter(torch.zeros(self.dim))

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        return embeddings + self.translation


class LinearOperator(AbstractOperator):

    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        # We add a dimension so that matmul performs a matrix-vector product.
        return torch.matmul(self.linear_transformation,
                            embeddings.unsqueeze(-1)).squeeze(-1)


class AffineOperator(AbstractOperator):

    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))
        self.translation = nn.Parameter(torch.zeros(self.dim))

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        # We add a dimension so that matmul performs a matrix-vector product.
        return (torch.matmul(self.linear_transformation,
                             embeddings.unsqueeze(-1)).squeeze(-1)
                + self.translation)

    # FIXME This adapts from the pre-D14024710 format; remove eventually.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = "%slinear_transformation" % prefix
        old_param_key = "%srotation" % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = \
                state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ComplexDiagonalOperator(AbstractOperator):

    def __init__(self, dim: int):
        super().__init__(dim)
        if dim % 2 != 0:
            raise ValueError("Need even dimension as 1st half is real "
                             "and 2nd half is imaginary coordinates")
        self.real = nn.Parameter(torch.ones(self.dim // 2))
        self.imag = nn.Parameter(torch.zeros(self.dim // 2))

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        real = embeddings[..., :self.dim // 2]
        imag = embeddings[..., self.dim // 2:]
        prod = torch.empty(*embeddings.size())
        prod[..., :self.dim // 2] = real * self.real - imag * self.imag
        prod[..., self.dim // 2:] = real * self.imag + imag * self.real
        return prod


OPERATORS: Dict[Operator, Type[AbstractOperator]] = {
    Operator.NONE: IdentityOperator,
    Operator.DIAGONAL: DiagonalOperator,
    Operator.TRANSLATION: TranslationOperator,
    Operator.LINEAR: LinearOperator,
    Operator.AFFINE: AffineOperator,
    Operator.COMPLEX_DIAGONAL: ComplexDiagonalOperator,
}


class AbstractDynamicOperator(nn.Module, ABC):

    """Perform different operations on many vectors.

    The inputs are a tensor containing a set of vectors and another tensor
    specifying, for each vector, which operation to apply to it. The output has
    the same size as the first input and contains the outputs of the operations
    applied to the input vectors. The different operations are identified by
    integers in a [0, N) range. They are all of the same type (say, translation)
    but each one has its own set of parameters. The dimension of the vectors and
    the total number of operations that need to be supported are provided at
    initialization. The first tensor can have any number of dimensions (>= 1).

    """

    def __init__(self, dim: int, num_operations: int):
        super().__init__()
        self.dim = dim
        self.num_operations = num_operations

    @abstractmethod
    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        pass


class IdentityDynamicOperator(AbstractDynamicOperator):

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings


class DiagonalDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.diagonals = nn.Parameter(torch.ones(self.num_operations, self.dim))

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return self.diagonals[operator_idxs] * embeddings


class TranslationDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.translations = nn.Parameter(torch.zeros(self.num_operations, self.dim))

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings + self.translations[operator_idxs]


class LinearDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(
            torch.diag_embed(torch.ones(()).expand(num_operations, dim)))

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        # We add a dimension so that matmul performs a matrix-vector product.
        return torch.matmul(self.linear_transformations[operator_idxs],
                            embeddings.unsqueeze(-1)).squeeze(-1)


class AffineDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(
            torch.diag_embed(torch.ones(()).expand(num_operations, dim)))
        self.translations = nn.Parameter(torch.zeros(self.num_operations, self.dim))

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        # We add a dimension so that matmul performs a matrix-vector product.
        return (torch.matmul(self.linear_transformations[operator_idxs],
                             embeddings.unsqueeze(-1)).squeeze(-1)
                + self.translations[operator_idxs])

    # FIXME This adapts from the pre-D14024710 format; remove eventually.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = "%slinear_transformations" % prefix
        old_param_key = "%srotations" % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = \
                state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ComplexDiagonalDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        if dim % 2 != 0:
            raise ValueError("Need even dimension as 1st half is real "
                             "and 2nd half is imaginary coordinates")
        self.real = nn.Parameter(torch.ones(self.num_operations, self.dim // 2))
        self.imag = nn.Parameter(torch.zeros(self.num_operations, self.dim // 2))

    def forward(
        self,
        embeddings: torch.FloatTensor,
        operator_idxs: torch.LongTensor,
    ) -> torch.FloatTensor:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        real_a = embeddings[..., :self.dim // 2]
        imag_a = embeddings[..., self.dim // 2:]
        real_b = self.real[operator_idxs]
        imag_b = self.imag[operator_idxs]
        prod = torch.empty(*embeddings.size())
        prod[..., :self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2:] = real_a * imag_b + imag_a * real_b
        return prod


DYNAMIC_OPERATORS: Dict[Operator, Type[AbstractDynamicOperator]] = {
    Operator.NONE: IdentityDynamicOperator,
    Operator.DIAGONAL: DiagonalDynamicOperator,
    Operator.TRANSLATION: TranslationDynamicOperator,
    Operator.LINEAR: LinearDynamicOperator,
    Operator.AFFINE: AffineDynamicOperator,
    Operator.COMPLEX_DIAGONAL: ComplexDiagonalDynamicOperator,
}


def instantiate_operator(
    operator: Operator,
    side: Side,
    num_dynamic_rels: int,
    dim: int,
) -> AbstractOperator:
    if num_dynamic_rels > 0:
        try:
            dynamic_operator_class = DYNAMIC_OPERATORS[operator]
        except KeyError:
            raise NotImplementedError(
                "Unknown operator for dynamic rels: %s" % operator)
        return dynamic_operator_class(dim, num_dynamic_rels)
    elif side is Side.LHS:
        return IdentityOperator(dim)
    else:
        try:
            operator_class = OPERATORS[operator]
        except KeyError:
            raise NotImplementedError("Unknown operator: %s" % operator)
        return operator_class(dim)


class AbstractComparator(nn.Module, ABC):

    """Calculate scores between pairs of given vectors in a certain space.

    The input consists of four tensors each representing a set of vectors: one
    set for each pair of the product between <left-hand side vs right-hand side>
    and <positive vs negative>. Each of these sets is chunked into the same
    number of chunks. The chunks have all the same size within each set, but
    different sets may have chunks of different sizes (except the two positive
    sets, which have chunks of the same size). All the vectors have the same
    number of dimensions. In short, the four tensor have these sizes:

        L+: C x P x D     R+: C x P x D     L-: C x L x D     R-: C x R x D

    The output consists of three tensors:
    - One for the scores between the corresponding pairs in L+ and R+. That is,
      for each chunk on one side, each vector of that chunk is compared only
      with the corresponding vector in the corresponding chunk on the other
      side. Think of it as the "inner" product of the two sides, or a matching.
    - Two for the scores between R+ and L- and between L+ and R-, where for each
      pair of corresponding chunks, all the vectors on one side are compared
      with all the vectors on the other side. Think of it as a per-chunk "outer"
      product, or a complete bipartite graph.
    Hence the sizes of the three output tensors are:

        ⟨L+,R+⟩: C x P     R+ ⊗ L-: C x P x L     L+ ⊗ R-: C x P x R

    Some comparators may need to peform a certain operation in the same way on
    all input vectors (say, normalizing them) before starting to compare them.
    When some vectors are used as both positives and negatives, the operation
    should ideally only be performed once. For that to occur, comparators expose
    a prepare method that the user should call on the vectors before passing
    them to the forward method, taking care of calling it only once on
    duplicated inputs.

    """

    @abstractmethod
    def prepare(
        self,
        embs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pass

    @abstractmethod
    def forward(
        self,
        lhs_pos: torch.FloatTensor,
        rhs_pos: torch.FloatTensor,
        lhs_neg: torch.FloatTensor,
        rhs_neg: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pass


class DotComparator(AbstractComparator):

    def prepare(
        self,
        embs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return embs

    def forward(
        self,
        lhs_pos: torch.FloatTensor,
        rhs_pos: torch.FloatTensor,
        lhs_neg: torch.FloatTensor,
        rhs_neg: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)

        # Equivalent to (but faster than) torch.einsum('cid,cid->ci', ...).
        pos_scores = (lhs_pos * rhs_pos).sum(-1)
        # Equivalent to (but faster than) torch.einsum('cid,cjd->cij', ...).
        lhs_neg_scores = torch.bmm(rhs_pos, lhs_neg.transpose(-1, -2))
        rhs_neg_scores = torch.bmm(lhs_pos, rhs_neg.transpose(-1, -2))

        return pos_scores, lhs_neg_scores, rhs_neg_scores


class CosComparator(AbstractComparator):

    def prepare(
        self,
        embs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Dividing by the norm costs N * dim divisions, multiplying by the
        # reciprocal of the norm costs N divisions and N * dim multiplications.
        # The latter one is faster.
        norm = embs.norm(2, dim=-1)
        return embs * norm.reciprocal().unsqueeze(-1)

    def forward(
        self,
        lhs_pos: torch.FloatTensor,
        rhs_pos: torch.FloatTensor,
        lhs_neg: torch.FloatTensor,
        rhs_neg: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)

        # Equivalent to (but faster than) torch.einsum('cid,cid->ci', ...).
        pos_scores = (lhs_pos * rhs_pos).sum(-1)
        # Equivalent to (but faster than) torch.einsum('cid,cjd->cij', ...).
        lhs_neg_scores = torch.bmm(rhs_pos, lhs_neg.transpose(-1, -2))
        rhs_neg_scores = torch.bmm(lhs_pos, rhs_neg.transpose(-1, -2))

        return pos_scores, lhs_neg_scores, rhs_neg_scores


class BiasedComparator(AbstractComparator):

    def __init__(self, base_comparator):
        super().__init__()
        self.base_comparator = base_comparator

    def prepare(
        self,
        embs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.cat([embs[..., :1], self.base_comparator.prepare(embs[..., 1:])], dim=-1)

    def forward(
        self,
        lhs_pos: torch.FloatTensor,
        rhs_pos: torch.FloatTensor,
        lhs_neg: torch.FloatTensor,
        rhs_neg: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)

        pos_scores, lhs_neg_scores, rhs_neg_scores = self.base_comparator.forward(
            lhs_pos[..., 1:], rhs_pos[..., 1:], lhs_neg[..., 1:], rhs_neg[..., 1:])

        lhs_pos_bias = lhs_pos[..., 0]
        rhs_pos_bias = rhs_pos[..., 0]

        pos_scores += lhs_pos_bias
        pos_scores += rhs_pos_bias

        lhs_neg_scores += rhs_pos_bias.unsqueeze(-1)
        lhs_neg_scores += lhs_neg[..., 0].unsqueeze(-2)

        rhs_neg_scores += lhs_pos_bias.unsqueeze(-1)
        rhs_neg_scores += rhs_neg[..., 0].unsqueeze(-2)

        return pos_scores, lhs_neg_scores, rhs_neg_scores


class AbstractLoss(nn.Module, ABC):

    """Calculate weighted loss of scores for positive and negative pairs.

    The inputs are a 1-D tensor of size P containing scores for positive pairs
    of entities (i.e., those among which an edge exists) and a P x N tensor
    containing scores for negative pairs (i.e., where no edge should exist). The
    pairs of entities corresponding to pos_scores[i] and to neg_scores[i,j] have
    at least one endpoint in common. The output is the loss value these scores
    induce and the margin for each negative score, which is zero if the negative
    score is smaller than the positive one by exactly the minimum expected
    amount, larger than zero if it's closer to the positive and smaller than
    zero if it's farther away. The margin will be returned only for the ranking
    loss, and will be zero for all other functions. If the method supports
    weighting (as is the case for the logistic loss) all positive scores will be
    weighted by the same weight and so will all the negative ones.

    """

    @abstractmethod
    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass


class LogisticLoss(AbstractLoss):

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)
        neg_weight = 1 / num_neg if num_neg > 0 else 0

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.tensor(1.).expand(num_pos),
            reduction='sum',
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.tensor(0.).expand(num_pos, num_neg),
            reduction='sum',
        )

        loss = pos_loss + neg_weight * neg_loss
        margin = torch.tensor(0.).expand(num_pos, num_neg)

        return loss, margin


class RankingLoss(AbstractLoss):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)

        # FIXME Workaround for https://github.com/pytorch/pytorch/issues/15223.
        if num_pos == 0 or num_neg == 0:
            return torch.tensor(0., requires_grad=True), torch.empty(num_pos, num_neg)

        margin = neg_scores - pos_scores.unsqueeze(1) + self.margin
        loss = margin.clamp(min=0).sum()

        return loss, margin


class SoftmaxLoss(AbstractLoss):

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)

        # FIXME Workaround for https://github.com/pytorch/pytorch/issues/15870
        # and https://github.com/pytorch/pytorch/issues/15223.
        if num_pos == 0 or num_neg == 0:
            return torch.tensor(0., requires_grad=True), torch.empty(num_pos, num_neg)

        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        loss = F.cross_entropy(
            scores,
            torch.tensor(0).expand(num_pos),
            reduction='sum',
        )
        margin = torch.tensor(0.).expand(num_pos, num_neg)

        return loss, margin


class Negatives(Enum):
    NONE = "none"
    UNIFORM = "uniform"
    BATCH_UNIFORM = "batch_uniform"
    ALL = "all"


Mask = List[Tuple[Union[int, slice, Sequence[int], torch.LongTensor], ...]]

# lhs_margin, rhs_margin
Margins = Tuple[torch.FloatTensor, torch.FloatTensor]

# lhs_pos, rhs_pos, lhs_neg, rhs_neg
Scores = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]


class MultiRelationEmbedder(nn.Module):
    """
    A multi-relation embedding model.

    Graph embedding on multiple relations over multiple entity types. Each
    relation consists of a lhs and rhs entity type, and optionally a relation
    operator (which is a learned multiplicative vector - see e.g.
    https://arxiv.org/abs/1510.04935)

    The model includes the logic for training using a ranking loss over a mixture
    of negatives sampled from the batch and uniformly from the entities. An
    optimization is used for negative sampling, where each batch is divided into
    sub-batches of size num_batch_negs, which are used as negative samples against
    each other. Each of these sub-batches also receives num_uniform_negs (common)
    negative samples sampled uniformly from the entities of the lhs and rhs types.
    """

    # A ModuleDict is used to store embeddings for entities, indexed by name.
    # As items are also attributes, we need to prefix them to avoid collisions.
    EMB_PREFIX = "emb_"

    def __init__(
        self,
        dim: int,
        relations: List[RelationSchema],
        entities: Dict[str, EntitySchema],
        num_batch_negs: int,
        num_uniform_negs: int,
        margin: float = 0.1,
        comparator: Comparator = Comparator.COS,
        global_emb: bool = False,
        max_norm: Optional[float] = None,
        loss_fn: LossFunction = LossFunction.RANKING,
        bias: bool = False,
        num_dynamic_rels: int = 0,
    ):

        super(MultiRelationEmbedder, self).__init__()

        self.dim: int = dim

        self.relations: List[RelationSchema] = relations
        self.entities: Dict[str, EntitySchema] = entities
        self.num_dynamic_rels: int = num_dynamic_rels
        if num_dynamic_rels > 0:
            assert len(relations) == 1
            assert len(entities) == 1
        self.lhs_operators: nn.ModuleList = nn.ModuleList()
        self.rhs_operators: nn.ModuleList = nn.ModuleList()
        for r in relations:
            self.lhs_operators.append(
                instantiate_operator(r.operator, Side.LHS, num_dynamic_rels, dim))
            self.rhs_operators.append(
                instantiate_operator(r.operator, Side.RHS, num_dynamic_rels, dim))

        self.num_batch_negs: int = num_batch_negs
        self.num_uniform_negs: int = num_uniform_negs

        if comparator is Comparator.DOT:
            self.comparator = DotComparator()
        elif comparator is Comparator.COS:
            self.comparator = CosComparator()
        else:
            raise NotImplementedError("Unknown comparator: %s" % comparator)

        if bias:
            self.comparator = BiasedComparator(self.comparator)

        if loss_fn is LossFunction.LOGISTIC:
            self.loss_fn = LogisticLoss()
        elif loss_fn is LossFunction.RANKING:
            self.loss_fn = RankingLoss(margin)
        elif loss_fn is LossFunction.SOFTMAX:
            self.loss_fn = SoftmaxLoss()
        else:
            raise NotImplementedError("Unknown loss function: %s" % loss_fn)

        if loss_fn is LossFunction.LOGISTIC and comparator is Comparator.COS:
            log("WARNING: You have logistic loss and cosine distance. Are you sure?")

        self.lhs_embs: nn.ParameterDict = nn.ModuleDict()
        self.rhs_embs: nn.ParameterDict = nn.ModuleDict()

        if global_emb:
            self.global_embs: Optional[nn.ParameterDict] = nn.ParameterDict()
            for entity in entities.keys():
                self.global_embs[self.EMB_PREFIX + entity] = \
                    nn.Parameter(torch.zeros(dim))
        else:
            self.global_embs: Optional[nn.ParameterDict] = None

        self.max_norm: Optional[float] = max_norm

    def set_embeddings(self, entity: str, weights: nn.Parameter, side: Side):
        if self.entities[entity].featurized:
            emb = FeaturizedEmbedding(weights, max_norm=self.max_norm)
        else:
            emb = SimpleEmbedding(weights, max_norm=self.max_norm)
        side.pick(self.lhs_embs, self.rhs_embs)[self.EMB_PREFIX + entity] = emb

    def clear_embeddings(self, entity: str, side: Side) -> None:
        embs = side.pick(self.lhs_embs, self.rhs_embs)
        try:
            del embs[self.EMB_PREFIX + entity]
        except KeyError:
            pass

    def get_embeddings(self, entity: str, side: Side) -> nn.Parameter:
        embs = side.pick(self.lhs_embs, self.rhs_embs)
        try:
            emb = embs[self.EMB_PREFIX + entity]
        except KeyError:
            return None
        else:
            return emb.weight

    def get_relation_parameters(self):
        if self.num_dynamic_rels > 0:
            rels_lhs = next(self.lhs_operators[0].parameters())
            rels_rhs = next(self.rhs_operators[0].parameters())
            return rels_lhs, rels_rhs
        else:
            rels = []
            for operator in self.rhs_operators:
                if len(list(operator.parameters())) == 0:
                    rels.append(None)
                    continue
                rels.append(
                    torch.cat([
                        parameter.view(-1, self.dim)
                        for parameter in operator.parameters()
                    ], dim=0).squeeze()
                )

    def adjust_embs(
        self,
        embs: torch.FloatTensor,
        rel: Union[int, Optional[torch.LongTensor]],
        side: Side,
    ) -> torch.FloatTensor:

        # 1. Apply the global embedding, if enabled
        if self.global_embs is not None:
            if not isinstance(rel, int):
                raise RuntimeError("Cannot have global embs with dynamic rels")
            relation = self.relations[rel]
            entity = side.pick(relation.lhs, relation.rhs)
            embs += self.global_embs[self.EMB_PREFIX + entity]

        # 2. Apply the relation operator
        if self.num_dynamic_rels > 0:
            if rel is not None:
                embs = side.pick(self.lhs_operators, self.rhs_operators)[0](embs, rel)
        else:
            embs = side.pick(self.lhs_operators, self.rhs_operators)[rel](embs)

        # 3. Prepare for the comparator.
        embs = self.comparator.prepare(embs)

        return embs

    def is_featurized(self, rel, side: Side):
        if self.num_dynamic_rels > 0:
            rel = 0
        rel_config = self.relations[rel]
        ent = side.pick(rel_config.lhs, rel_config.rhs)
        return self.entities[ent].featurized

    def prepare_negatives(
        self,
        pos_input: Union[torch.LongTensor, TensorList],
        pos_embs: torch.FloatTensor,
        module: AbstractEmbedding,
        type_: Negatives,
        num_uniform_neg: int,
        *,
        rel: Union[int, Optional[torch.LongTensor]],
        side: Side,
    ) -> Tuple[torch.FloatTensor, Mask]:
        """Given some chunked positives, set up chunks of negatives.

        This function operates on one side (left-hand or right-hand) at a time.
        It takes all the information about the positives on that side (the
        original input value, the corresponding embeddings, and the module used
        to convert one to the other). It then produces negatives for that side
        according to the specified mode. The positive embeddings come in in
        chunked form and the negatives are produced within each of these chunks.
        The negatives can be either none, or the positives from the same chunk,
        or all the possible entities. In the second mode, uniformly-sampled
        entities can also be appended to the per-chunk negatives (each chunk
        having a different sample). This function returns both the chunked
        embeddings of the negatives and a mask of the same size as the chunked
        positives-vs-negatives scores, whose non-zero elements correspond to the
        scores that must be ignored.

        """
        num_pos = match_shape(pos_input, -1)
        num_chunks, chunk_size, dim = match_shape(pos_embs, -1, -1, -1)
        last_chunk_size = num_pos - (num_chunks - 1) * chunk_size

        ignore_mask: Mask = []
        if type_ is Negatives.NONE:
            neg_embs = torch.empty(num_chunks, 0, dim)
        elif type_ is Negatives.UNIFORM:
            neg_embs = module.sample_entities(
                num_chunks, num_uniform_neg)
        elif type_ is Negatives.BATCH_UNIFORM:
            neg_embs = pos_embs
            if num_uniform_neg > 0:
                try:
                    uniform_neg_embs = module.sample_entities(
                        num_chunks, num_uniform_neg)
                except NotImplementedError:
                    pass  # only use pos_embs i.e. batch negatives
                else:
                    neg_embs = torch.cat([
                        pos_embs,
                        self.adjust_embs(uniform_neg_embs, rel=rel, side=side)
                    ], dim=1)

            chunk_indices = torch.arange(chunk_size)
            last_chunk_indices = chunk_indices[:last_chunk_size]
            # Ignore scores between positive pairs.
            ignore_mask.append(
                (slice(num_chunks - 1), chunk_indices, chunk_indices))
            ignore_mask.append((-1, last_chunk_indices, last_chunk_indices))
            # In the last chunk, ignore the scores between the positives that
            # are not padding (i.e., the first last_chunk_size ones) and the
            # negatives that are padding (i.e., all of them except the first
            # last_chunk_size ones). Stop the last slice at chunk_size so that
            # it doesn't also affect the uniformly-sampled negatives.
            ignore_mask.append(
                (-1, slice(last_chunk_size), slice(last_chunk_size, chunk_size)))

        elif type_ is Negatives.ALL:
            if not isinstance(pos_input, torch.LongTensor):
                raise TypeError("Cannot use all entities as negatives "
                                "without the IDs of the positive entities")
            neg_embs = self.adjust_embs(
                module.get_all_entities().expand(num_chunks, -1, dim),
                rel=rel, side=side,
            )

            if num_uniform_neg > 0:
                log("WARNING: Adding uniform negatives makes no sense "
                    "when already using all negatives")

            chunk_indices = torch.arange(chunk_size)
            last_chunk_indices = chunk_indices[:last_chunk_size]
            # Ignore scores between positive pairs: since the i-th such pair has
            # the pos_input[i] entity on this side, ignore_mask[i, pos_input[i]]
            # must be set to 1 for every i. This becomes slightly more tricky as
            # the rows may be wrapped into multiple chunks (the last of which
            # may be smaller).
            ignore_mask.append((
                torch.arange(num_chunks - 1).unsqueeze(1),
                chunk_indices.unsqueeze(0),
                pos_input[:-last_chunk_size].view(num_chunks - 1, chunk_size),
            ))
            ignore_mask.append(
                (-1, last_chunk_indices, pos_input[-last_chunk_size:]))

        else:
            raise NotImplementedError("Unknown negative type %s" % type_)

        return neg_embs, ignore_mask

    def forward(
        self,
        lhs: Union[torch.LongTensor, TensorList],
        rhs: Union[torch.LongTensor, TensorList],
        rel: Union[int, torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, Margins, Scores]:
        num_pos = match_shape(lhs, -1)
        match_shape(rhs, num_pos)

        chunk_size: int
        lhs_negatives: Negatives
        lhs_num_uniform_negs: int
        rhs_negatives: Negatives
        rhs_num_uniform_negs: int

        if self.num_dynamic_rels > 0:
            if not isinstance(rel, torch.LongTensor):
                raise TypeError("Need relation for each positive pair")
            match_shape(rel, num_pos)
            relation_idx = 0
        else:
            if not isinstance(rel, int):
                raise TypeError("All positive pairs must come from the same relation")
            relation_idx = rel

        relation = self.relations[relation_idx]
        lhs_module = self.lhs_embs[self.EMB_PREFIX + relation.lhs]
        rhs_module = self.rhs_embs[self.EMB_PREFIX + relation.rhs]
        lhs_pos = lhs_module(lhs)
        rhs_pos = rhs_module(rhs)

        if self.num_dynamic_rels == 0:
            # In this case the operator is only applied to the RHS. This means
            # that an edge (u, r, v) is scored with m(u, f_r(v)), whereas the
            # negatives (u', r, v) and (u, r, v') are scored respectively with
            # m(u', f_r(v)) and m(u, f_r(v')). Since r is always the same, each
            # positive and negative right-hand side entity is only passed once
            # through the operator.
            lhs_pos = self.adjust_embs(lhs_pos, rel=rel, side=Side.LHS)
            rhs_pos = self.adjust_embs(rhs_pos, rel=rel, side=Side.RHS)

        else:
            # In this case the positive edges may come from different relations.
            # This makes it inefficient to apply the operators to the negatives
            # in the way we do above, because for a negative edge (u, r, v') we
            # would need to compute f_r(v'), with r being different from the one
            # in any positive pair that has v' on the right-hand side, which
            # could lead to v being passed through many different (potentially
            # all) operators. This would result in a combinatorial explosion.
            # So, instead, we duplicate all operators, creating two versions of
            # them, one for each side, and only allow one of them to be applied
            # at any given time. The edge (u, r, v) can thus be scored in two
            # ways, either as m(g_r(u), v) or as m(u, h_r(v)). The negatives
            # (u', r, v) and (u, r, v') are scored respectively as m(u', h_r(v))
            # and m(g_r(u), v'). This way we only need to perform two operator
            # applications for every positive input edge, one for each side.
            lhs_r_pos = self.adjust_embs(lhs_pos, rel=rel, side=Side.LHS)
            rhs_r_pos = self.adjust_embs(rhs_pos, rel=rel, side=Side.RHS)
            lhs_pos = self.adjust_embs(lhs_pos, rel=None, side=Side.LHS)
            rhs_pos = self.adjust_embs(rhs_pos, rel=None, side=Side.RHS)

        if relation.all_negs:
            chunk_size = num_pos
            negative_sampling_method = Negatives.ALL
        elif self.num_batch_negs == 0:
            chunk_size = self.num_uniform_negs
            negative_sampling_method = Negatives.UNIFORM
        else:
            chunk_size = self.num_batch_negs
            negative_sampling_method = Negatives.BATCH_UNIFORM

        num_chunks = (num_pos - 1) // chunk_size + 1  # ceil(num_pos / chunk_size)
        if num_pos < num_chunks * chunk_size:
            padding = torch.tensor(0.).expand(num_chunks * chunk_size - num_pos, self.dim)
            lhs_pos = torch.cat([lhs_pos, padding], dim=0)
            rhs_pos = torch.cat([rhs_pos, padding], dim=0)
            if self.num_dynamic_rels > 0:
                lhs_r_pos = torch.cat([lhs_r_pos, padding], dim=0)
                rhs_r_pos = torch.cat([rhs_r_pos, padding], dim=0)
        lhs_pos = lhs_pos.view(num_chunks, chunk_size, self.dim)
        rhs_pos = rhs_pos.view(num_chunks, chunk_size, self.dim)
        if self.num_dynamic_rels > 0:
            lhs_r_pos = lhs_r_pos.view(num_chunks, chunk_size, self.dim)
            rhs_r_pos = rhs_r_pos.view(num_chunks, chunk_size, self.dim)

        if self.num_dynamic_rels == 0:
            lhs_neg, lhs_ignore_mask = self.prepare_negatives(
                lhs, lhs_pos, lhs_module, negative_sampling_method,
                self.num_uniform_negs, rel=rel, side=Side.LHS)
            rhs_neg, rhs_ignore_mask = self.prepare_negatives(
                rhs, rhs_pos, rhs_module, negative_sampling_method,
                self.num_uniform_negs, rel=rel, side=Side.RHS)
            pos_scores, lhs_neg_scores, rhs_neg_scores = self.comparator(
                lhs_pos, rhs_pos, lhs_neg, rhs_neg)
            lhs_pos_scores = rhs_pos_scores = pos_scores

        else:
            lhs_neg, lhs_ignore_mask = self.prepare_negatives(
                lhs, lhs_pos, lhs_module, negative_sampling_method,
                self.num_uniform_negs, rel=None, side=Side.LHS)
            rhs_neg, rhs_ignore_mask = self.prepare_negatives(
                rhs, rhs_pos, rhs_module, negative_sampling_method,
                self.num_uniform_negs, rel=None, side=Side.RHS)
            lhs_pos_scores, lhs_neg_scores, _ = self.comparator(
                lhs_pos, rhs_r_pos, lhs_neg, torch.empty(num_chunks, 0, self.dim))
            rhs_pos_scores, _, rhs_neg_scores = self.comparator(
                lhs_r_pos, rhs_pos, torch.empty(num_chunks, 0, self.dim), rhs_neg)

        # The masks tell us which negative scores (i.e., scores for non-existing
        # edges) must be ignored because they come from pairs we don't actually
        # intend to compare (say, positive pairs or interactions with padding).
        # We do it by replacing them with a "very negative" value so that they
        # are considered spot-on predictions with minimal impact on the loss.
        for ignore_mask in lhs_ignore_mask:
            lhs_neg_scores[ignore_mask] = -1e9
        for ignore_mask in rhs_ignore_mask:
            rhs_neg_scores[ignore_mask] = -1e9

        # De-chunk the scores and ignore the ones whose positives were padding.
        lhs_pos_scores = lhs_pos_scores.flatten(0, 1)[:num_pos]
        rhs_pos_scores = rhs_pos_scores.flatten(0, 1)[:num_pos]
        lhs_neg_scores = lhs_neg_scores.flatten(0, 1)[:num_pos]
        rhs_neg_scores = rhs_neg_scores.flatten(0, 1)[:num_pos]

        lhs_loss, lhs_margin = self.loss_fn(lhs_pos_scores, lhs_neg_scores)
        rhs_loss, rhs_margin = self.loss_fn(rhs_pos_scores, rhs_neg_scores)
        loss = relation.weight * (lhs_loss + rhs_loss)


        return loss, (lhs_margin, rhs_margin), \
            (lhs_pos_scores.unsqueeze(-1), rhs_pos_scores.unsqueeze(-1),
             lhs_neg_scores, rhs_neg_scores)


def make_model(
    config: ConfigSchema,
    num_dynamic_rels: int = 0,
) -> MultiRelationEmbedder:
    if config.dynamic_relations:
        if len(config.relations) != 1:
            raise RuntimeError(
                "Dynamic relations are enabled, so there should only be one "
                "entry in config.relations with config for all relations."
            )
        try:
            with open(os.path.join(config.entity_path, "dynamic_rel_count.txt"), "rt") as tf:
                num_dynamic_rels = int(tf.read().strip())
        except FileNotFoundError:
            raise RuntimeError(
                "Dynamic relations are enabled, so there should be a file called "
                "dynamic_rel_count.txt in the entity path with their count."
            )
    else:
        num_dynamic_rels = 0

    if config.num_batch_negs > 0 and config.batch_size % config.num_batch_negs != 0:
        raise RuntimeError(
            "Batch size (%d) must be a multiple of num_batch_negs (%d)" %
            (config.batch_size, config.num_batch_negs)
        )

    model = MultiRelationEmbedder(
        config.dimension,
        config.relations,
        config.entities,
        num_uniform_negs=config.num_uniform_negs,
        num_batch_negs=config.num_batch_negs,
        margin=config.margin,
        comparator=config.comparator,
        global_emb=config.global_emb,
        max_norm=config.max_norm,
        loss_fn=config.loss_fn,
        bias=config.bias,
        num_dynamic_rels=num_dynamic_rels,
    )
    model.share_memory()
    return model


@contextmanager
def override_model(model, **new_config):
    old_config = {k: getattr(model, k) for k in new_config}
    for k, v in new_config.items():
        setattr(model, k, v)
    yield
    for k, v in old_config.items():
        setattr(model, k, v)
