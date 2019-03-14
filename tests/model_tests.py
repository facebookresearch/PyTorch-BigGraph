#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# In order to keep values visually aligned in matrix form we use double spaces
# and exceed line length. Tell flake8 to tolerate that. Ideally we'd want to
# disable only those two checks but there doesn't seem to be a way to do so.
# flake8: noqa

from unittest import TestCase, main

import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.model import (
    match_shape,
    # Embeddings
    SimpleEmbedding, FeaturizedEmbedding,
    # Operators
    IdentityOperator, DiagonalOperator, TranslationOperator, LinearOperator,
    AffineOperator, ComplexDiagonalOperator,
    # Dynamic operators
    IdentityDynamicOperator, DiagonalDynamicOperator, TranslationDynamicOperator,
    LinearDynamicOperator, AffineDynamicOperator, ComplexDiagonalDynamicOperator,
    # Comparator
    DotComparator, CosComparator, BiasedComparator,
    # Losses
    LogisticLoss, RankingLoss, SoftmaxLoss,
)


def assertTensorEqual(self, actual, expected):
    if not isinstance(actual, torch.FloatTensor):
        self.fail("Expected FloatTensor, got %s" % type(actual))
    if actual.size() != expected.size():
        self.fail("Expected tensor of size %s, got %s"
                  % (expected.size(), actual.size()))
    if not torch.allclose(actual, expected, rtol=0.00005, atol=0.00005, equal_nan=True):
        self.fail("Expected\n%r\ngot\n%r" % (expected, actual))


class TestMatchShape(TestCase):

    def test_zero_dimensions(self):
        t = torch.tensor(42)
        self.assertIsNone(match_shape(t))
        self.assertIsNone(match_shape(t, ...))
        with self.assertRaises(TypeError):
            match_shape(t, 0)
        with self.assertRaises(TypeError):
            match_shape(t, 1)
        with self.assertRaises(TypeError):
            match_shape(t, -1)

    def test_one_dimension(self):
        t = torch.zeros(3)
        self.assertIsNone(match_shape(t, 3))
        self.assertIsNone(match_shape(t, ...))
        self.assertIsNone(match_shape(t, 3, ...))
        self.assertIsNone(match_shape(t, ..., 3))
        self.assertEqual(match_shape(t, -1), 3)
        with self.assertRaises(TypeError):
            match_shape(t)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 1)
        with self.assertRaises(TypeError):
            match_shape(t, 3, ..., 3)

    def test_many_dimension(self):
        t = torch.zeros(3, 4, 5)
        self.assertIsNone(match_shape(t, 3, 4, 5))
        self.assertIsNone(match_shape(t, ...))
        self.assertIsNone(match_shape(t, ..., 5))
        self.assertIsNone(match_shape(t, 3, ..., 5))
        self.assertIsNone(match_shape(t, 3, 4, 5, ...))
        self.assertEqual(match_shape(t, -1, 4, 5), 3)
        self.assertEqual(match_shape(t, -1, ...), 3)
        self.assertEqual(match_shape(t, -1, 4, ...), 3)
        self.assertEqual(match_shape(t, -1, ..., 5), 3)
        self.assertEqual(match_shape(t, -1, 4, -1), (3, 5))
        self.assertEqual(match_shape(t, ..., -1, -1), (4, 5))
        self.assertEqual(match_shape(t, -1, -1, -1), (3, 4, 5))
        self.assertEqual(match_shape(t, -1, -1, ..., -1), (3, 4, 5))
        with self.assertRaises(TypeError):
            match_shape(t)
        with self.assertRaises(TypeError):
            match_shape(t, 3)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4)
        with self.assertRaises(TypeError):
            match_shape(t, 5, 4, 3)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4, 5, 6)
        with self.assertRaises(TypeError):
            match_shape(t, 3, 4, ..., 4, 5)

    def test_bad_args(self):
        t = torch.tensor([])
        with self.assertRaises(RuntimeError):
            match_shape(t, ..., ...)
        with self.assertRaises(RuntimeError):
            match_shape(t, "foo")
        with self.assertRaises(AttributeError):
            match_shape(None)


class TestSimpleEmbedding(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings)
        assertTensorEqual(self, module(torch.tensor([2, 0, 0])), torch.tensor([
            [3., 3., 3.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]))

    def test_max_norm(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings, max_norm=2)
        assertTensorEqual(self, module(torch.tensor([2, 0, 0])), torch.tensor([
            [1.1547, 1.1547, 1.1547],
            [1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000],
        ]))

    def test_empty(self):
        embeddings = torch.empty(0, 3)
        module = SimpleEmbedding(weight=embeddings)
        assertTensorEqual(self, module(torch.empty(0, dtype=torch.long)), torch.empty(0, 3))

    def test_get_all_entities(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings)
        assertTensorEqual(self, module.get_all_entities(), torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ]))

    def test_get_all_entities_max_norm(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings, max_norm=2)
        assertTensorEqual(self, module.get_all_entities(), torch.tensor([
            [1.0000, 1.0000, 1.0000],
            [1.1547, 1.1547, 1.1547],
            [1.1547, 1.1547, 1.1547],
        ]))

    def test_sample_entities(self):
        torch.manual_seed(42)
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings)
        assertTensorEqual(self, module.sample_entities(2, 2), torch.tensor([
            [[1., 1., 1.],
             [3., 3., 3.]],
            [[2., 2., 2.],
             [2., 2., 2.]],
        ]))

    def test_sample_entities_max_norm(self):
        torch.manual_seed(42)
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings, max_norm=2)
        assertTensorEqual(self, module.sample_entities(2, 2), torch.tensor([
            [[1.0000, 1.0000, 1.0000],
             [1.1547, 1.1547, 1.1547]],
            [[1.1547, 1.1547, 1.1547],
             [1.1547, 1.1547, 1.1547]],
        ]))


class TestFeaturizedEmbedding(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = FeaturizedEmbedding(weight=embeddings)
        assertTensorEqual(self, module(TensorList(
            torch.tensor([0, 1, 3, 6, 6]),
            torch.tensor([0, 2, 1, 0, 1, 0]),
        )), torch.tensor([
            [1.0000, 1.0000, 1.0000],
            [2.5000, 2.5000, 2.5000],
            [1.3333, 1.3333, 1.3333],
            [0.0000, 0.0000, 0.0000],
        ]))

    def test_max_norm(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = FeaturizedEmbedding(weight=embeddings, max_norm=2)
        assertTensorEqual(self, module(TensorList(
            torch.tensor([0, 1, 3, 6, 6]),
            torch.tensor([0, 2, 1, 0, 1, 0]),
        )), torch.tensor([
            [1.0000, 1.0000, 1.0000],
            [1.1547, 1.1547, 1.1547],
            [1.0516, 1.0516, 1.0516],
            [0.0000, 0.0000, 0.0000],
        ]))

    def test_empty(self):
        embeddings = torch.empty(0, 3)
        module = FeaturizedEmbedding(weight=embeddings)
        assertTensorEqual(self, module(TensorList(torch.tensor([0]), torch.empty(0))), torch.empty(0, 3))

    def test_get_all_entities(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = FeaturizedEmbedding(weight=embeddings)
        with self.assertRaises(NotImplementedError):
            module.get_all_entities()

    def test_sample_entities(self):
        torch.manual_seed(42)
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = FeaturizedEmbedding(weight=embeddings)
        with self.assertRaises(NotImplementedError):
            module.sample_entities(2, 2)


class TestIdentityOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = IdentityOperator(3)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ]))


class TestDiagonalOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = DiagonalOperator(3)
        with torch.no_grad():
            operator.diagonal += torch.arange(3, dtype=torch.float)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[0.3766, 1.9468, 1.5570],
             [0.1801, 0.3170, 1.3755]],
            [[0.6188, 0.3834, 0.3018],
             [0.3876, 1.4268, 2.3763]],
        ]))


class TestTranslationOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = TranslationOperator(3)
        with torch.no_grad():
            operator.translation += torch.arange(3, dtype=torch.float)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[0.3766, 1.9734, 2.5190],
             [0.1801, 1.1585, 2.4585]],
            [[0.6188, 1.1917, 2.1006],
             [0.3876, 1.7134, 2.7921]],
        ]))


class TestLinearOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = LinearOperator(3)
        with torch.no_grad():
            operator.linear_transformation += torch.arange(9, dtype=torch.float).view(3, 3)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[ 2.3880,  8.5918, 13.7444],
             [ 1.2556,  3.6253,  6.3166]],
            [[ 1.0117,  3.3179,  5.9601],
             [ 2.6852,  8.6903, 14.4483]],
        ]))


class TestAffineOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = AffineOperator(3)
        with torch.no_grad():
            operator.linear_transformation += torch.arange(9, dtype=torch.float).view(3, 3)
            operator.translation += torch.arange(3, dtype=torch.float)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[ 2.3880,  9.5918, 15.7444],
             [ 1.2556,  4.6253,  8.3166]],
            [[ 1.0117,  4.3179,  7.9601],
             [ 2.6852,  9.6903, 16.4483]],
        ]))


class TestComplexDiagonalOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190, 0.5453],
             [0.1801, 0.1585, 0.4585, 0.5928]],
            [[0.6188, 0.1917, 0.1006, 0.3378],
             [0.3876, 0.7134, 0.7921, 0.9434]],
        ])
        operator = ComplexDiagonalOperator(4)
        with torch.no_grad():
            operator.real[...] = torch.tensor([0.2949, 0.0029])
            operator.imag[...] = torch.tensor([0.4070, 0.1027])
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[-0.1002, -0.0532,  0.3063,  0.1015],
             [-0.1335, -0.0604,  0.2085,  0.0180]],
            [[ 0.1415, -0.0341,  0.2815,  0.0207],
             [-0.2081, -0.0948,  0.3913,  0.0760]],
        ]))


class TestIdentityDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = IdentityDynamicOperator(3, 5)
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ]))


class TestDiagonalDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = DiagonalDynamicOperator(3, 5)
        with torch.no_grad():
            operator.diagonals += torch.arange(15, dtype=torch.float).view(5, 3)
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[0.3766, 1.9468, 1.5570],
             [2.3413, 2.2190, 6.8775]],
            [[4.3316, 1.5336, 0.9054],
             [0.3876, 1.4268, 2.3763]],
        ]))


class TestTranslationDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = TranslationDynamicOperator(3, 5)
        with torch.no_grad():
            operator.translations += torch.arange(15, dtype=torch.float).view(5, 3)
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[ 0.3766,  1.9734,  2.5190],
             [12.1801, 13.1585, 14.4585]],
            [[ 6.6188,  7.1917,  8.1006],
             [ 0.3876,  1.7134,  2.7921]],
        ]))


class TestLinearDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = LinearDynamicOperator(3, 5)
        with torch.no_grad():
            operator.linear_transformations += torch.arange(45, dtype=torch.float).view(5, 3, 3)
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[ 2.3880,  8.5918, 13.7444],
             [29.9512, 32.3209, 35.0122]],
            [[17.4115, 19.7177, 22.3599],
             [ 2.6852,  8.6903, 14.4483]],
        ]))


class TestAffineDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ])
        operator = AffineDynamicOperator(3, 5)
        with torch.no_grad():
            operator.linear_transformations += torch.arange(45, dtype=torch.float).view(5, 3, 3)
            operator.translations += torch.arange(15, dtype=torch.float).view(5, 3)
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[ 2.3880,  9.5918, 15.7444],
             [41.9512, 45.3209, 49.0122]],
            [[23.4115, 26.7177, 30.3599],
             [ 2.6852,  9.6903, 16.4483]],
        ]))


class TestComplexDiagonalDynamicOperator(TestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190, 0.5453],
             [0.1801, 0.1585, 0.4585, 0.5928]],
            [[0.6188, 0.1917, 0.1006, 0.3378],
             [0.3876, 0.7134, 0.7921, 0.9434]],
        ])
        operator = ComplexDiagonalDynamicOperator(4, 5)
        with torch.no_grad():
            operator.real[...] = torch.tensor([
                [0.2949, 0.0029],
                [0.5445, 0.5274],
                [0.3355, 0.9640],
                [0.6218, 0.2306],
                [0.3800, 0.9428],
            ])
            operator.imag[...] = torch.tensor([
                [0.4070, 0.1027],
                [0.1573, 0.0771],
                [0.4910, 0.1931],
                [0.3972, 0.4966],
                [0.9878, 0.2182],
            ])
        assertTensorEqual(self, operator(embeddings, torch.tensor([[0, 4], [2, 0]])), torch.tensor([
            [[-0.1002, -0.0532,  0.3063,  0.1015],
             [-0.3845,  0.0201,  0.3521,  0.5935]],
            [[ 0.1582,  0.1196,  0.3376,  0.3627],
             [-0.2081, -0.0948,  0.3913,  0.0760]],
        ]))


class TestDotComparator(TestCase):

    def test_forward_one_batch(self):
        comparator = DotComparator()
        lhs_pos = comparator.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.0000, 0.0000, 0.0000],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.0000, 0.0000, 0.0000],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [1.2024, 0.3246],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[0.6463, 1.4433, 0.0000, 1.1491],
             [0.5392, 0.7652, 0.0000, 0.5815]],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[1.0838, 0.0000, 0.6631, 0.3002],
             [0.9457, 0.0000, 0.6156, 0.2751]],
        ]))

    def test_forward_two_batches(self):
        comparator = DotComparator()
        lhs_pos = comparator.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.0000, 0.0000, 0.0000],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.0000, 0.0000, 0.0000]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [1.2024],
            [0.3246],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[0.6463, 1.4433]],
            [[0.0000, 0.5815]],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[1.0838, 0.0000]],
            [[0.6156, 0.2751]],
        ]))


class TestCosComparator(TestCase):

    def test_forward_one_batch(self):
        comparator = CosComparator()
        lhs_pos = comparator.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [0.9741, 0.6106],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[0.6165, 0.8749, 0.9664, 0.8701],
             [0.9607, 0.8663, 0.7494, 0.8224]],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[0.8354, 0.6406, 0.6626, 0.6856],
             [0.9063, 0.7439, 0.7648, 0.7810]],
        ]))

    def test_forward_two_batches(self):
        comparator = CosComparator()
        lhs_pos = comparator.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [0.9741],
            [0.6106],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[0.6165, 0.8749]],
            [[0.7494, 0.8224]],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[0.8354, 0.6406]],
            [[0.7648, 0.7810]],
        ]))


class TestBiasedComparator(TestCase):

    def test_forward_one_batch(self):
        comparator = BiasedComparator(CosComparator())
        lhs_pos = comparator.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [2.8086, 1.5434],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[1.7830, 2.5800, 2.3283, 2.4269],
             [1.5172, 2.0194, 1.4850, 1.9369]]
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[2.5017, 2.0980, 2.1129, 1.9578],
             [2.2670, 1.8759, 1.8838, 1.7381]],
        ]))

    def test_forward_two_batches(self):
        comparator = BiasedComparator(CosComparator())
        lhs_pos = comparator.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = comparator.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = comparator.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = comparator.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            comparator(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
        assertTensorEqual(self, pos_scores, torch.tensor([
            [2.8086],
            [1.5434],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [[1.7830, 2.5800]],
            [[1.4850, 1.9369]],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [[2.5017, 2.0980]],
            [[1.8838, 1.7381]],
        ]))


class TestLogisticLoss(TestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = LogisticLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(4.2589))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_forward_good(self):
        pos_scores = torch.full((3,), +1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), -1e9, requires_grad=True)
        loss_fn = LogisticLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), +1e9, requires_grad=True)
        loss_fn = LogisticLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(6e9))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.full((3,), 0., requires_grad=True)
        neg_scores = torch.full((3, 0), 0., requires_grad=True)
        loss_fn = LogisticLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(2.0794))
        assertTensorEqual(self, margin, torch.full((3, 0), 0.))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.full((0,), 0., requires_grad=True)
        neg_scores = torch.full((0, 0), 0., requires_grad=True)
        loss_fn = LogisticLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.full((0, 0), 0.))
        loss.backward()


class TestRankingLoss(TestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = RankingLoss(1.)
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(13.4475))
        assertTensorEqual(self, margin, torch.tensor([
            [0.6256, 0.8392, 1.1805, 0.4367, 0.2817],
            [1.0475, 0.8361, 0.8882, 0.9682, 0.7426],
            [1.6363, 0.8522, 0.8161, 0.6538, 1.6428],
        ]))
        loss.backward()

    def test_forward_good(self):
        pos_scores = torch.full((3,), 2., requires_grad=True)
        neg_scores = torch.full((3, 5), 1., requires_grad=True)
        loss_fn = RankingLoss(1.)
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1., requires_grad=True)
        neg_scores = torch.full((3, 5), 0., requires_grad=True)
        loss_fn = RankingLoss(1.)
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(30.))
        assertTensorEqual(self, margin, torch.full((3, 5), 2.))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.full((3,), 0., requires_grad=True)
        neg_scores = torch.empty((3, 0), requires_grad=True)
        loss_fn = RankingLoss(1.)
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.empty((3, 0)))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.empty((0,), requires_grad=True)
        neg_scores = torch.empty((0, 3), requires_grad=True)
        loss_fn = RankingLoss(1.)
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.empty((0, 3)))
        loss.backward()


class TestSoftmaxLoss(TestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = SoftmaxLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(5.2513))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_forward_good(self):
        pos_scores = torch.full((3,), +1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), -1e9, requires_grad=True)
        loss_fn = SoftmaxLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), +1e9, requires_grad=True)
        loss_fn = SoftmaxLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(6e9))
        assertTensorEqual(self, margin, torch.full((3, 5), 0.))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.full((3,), 0., requires_grad=True)
        neg_scores = torch.empty((3, 0), requires_grad=True)
        loss_fn = SoftmaxLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.empty((3, 0)))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.empty((0,), requires_grad=True)
        neg_scores = torch.empty((0, 3), requires_grad=True)
        loss_fn = SoftmaxLoss()
        loss, margin = loss_fn(pos_scores, neg_scores)
        assertTensorEqual(self, loss, torch.tensor(0.))
        assertTensorEqual(self, margin, torch.empty((0, 3)))
        loss.backward()


if __name__ == '__main__':
    main()
