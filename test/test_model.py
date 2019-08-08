#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

# In order to keep values visually aligned in matrix form we use double spaces
# and exceed line length. Tell flake8 to tolerate that. Ideally we'd want to
# disable only those two checks but there doesn't seem to be a way to do so.
# flake8: noqa

from unittest import TestCase, main

import torch
from torch_extensions.tensorlist.tensorlist import TensorList

from torchbiggraph.entitylist import EntityList
from torchbiggraph.model import (
    AffineDynamicOperator,
    AffineOperator,
    BiasedComparator,
    ComplexDiagonalDynamicOperator,
    ComplexDiagonalOperator,
    CosComparator,
    DiagonalDynamicOperator,
    DiagonalOperator,
    DotComparator,
    FeaturizedEmbedding,
    IdentityDynamicOperator,
    IdentityOperator,
    L2Comparator,
    LinearDynamicOperator,
    LinearOperator,
    SimpleEmbedding,
    SquaredL2Comparator,
    TranslationDynamicOperator,
    TranslationOperator,
    match_shape,
)


class TensorTestCase(TestCase):

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
        t = torch.zeros(())
        self.assertIsNone(match_shape(t))
        self.assertIsNone(match_shape(t, ...))
        with self.assertRaises(TypeError):
            match_shape(t, 0)
        with self.assertRaises(TypeError):
            match_shape(t, 1)
        with self.assertRaises(TypeError):
            match_shape(t, -1)

    def test_one_dimension(self):
        t = torch.zeros((3,))
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
        t = torch.zeros((3, 4, 5))
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
        t = torch.empty((0,))
        with self.assertRaises(RuntimeError):
            match_shape(t, ..., ...)
        with self.assertRaises(RuntimeError):
            match_shape(t, "foo")
        with self.assertRaises(AttributeError):
            match_shape(None)


class TestSimpleEmbedding(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ], requires_grad=True)
        module = SimpleEmbedding(weight=embeddings)
        result = module(EntityList.from_tensor(torch.tensor([2, 0, 0])))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [3., 3., 3.],
                [1., 1., 1.],
                [1., 1., 1.],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad.to_dense() != 0).any())

    def test_max_norm(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings, max_norm=2)
        self.assertTensorEqual(
            module(EntityList.from_tensor(torch.tensor([2, 0, 0]))),
            torch.tensor([
                [1.1547, 1.1547, 1.1547],
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ]))

    def test_empty(self):
        embeddings = torch.empty((0, 3))
        module = SimpleEmbedding(weight=embeddings)
        self.assertTensorEqual(
            module(EntityList.from_tensor(torch.empty((0,), dtype=torch.long))),
            torch.empty((0, 3)))

    def test_get_all_entities(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = SimpleEmbedding(weight=embeddings)
        self.assertTensorEqual(
            module.get_all_entities(),
            torch.tensor([
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
        self.assertTensorEqual(
            module.get_all_entities(),
            torch.tensor([
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
        self.assertTensorEqual(
            module.sample_entities(2, 2),
            torch.tensor([
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
        self.assertTensorEqual(
            module.sample_entities(2, 2),
            torch.tensor([
                [[1.0000, 1.0000, 1.0000],
                 [1.1547, 1.1547, 1.1547]],
                [[1.1547, 1.1547, 1.1547],
                 [1.1547, 1.1547, 1.1547]],
            ]))


class TestFeaturizedEmbedding(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ], requires_grad=True)
        module = FeaturizedEmbedding(weight=embeddings)
        result = module(EntityList.from_tensor_list(TensorList(
            torch.tensor([0, 1, 3, 6, 6]),
            torch.tensor([0, 2, 1, 0, 1, 0]),
        )))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [1.0000, 1.0000, 1.0000],
                [2.5000, 2.5000, 2.5000],
                [1.3333, 1.3333, 1.3333],
                [0.0000, 0.0000, 0.0000],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad.to_dense() != 0).any())

    def test_max_norm(self):
        embeddings = torch.tensor([
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ])
        module = FeaturizedEmbedding(weight=embeddings, max_norm=2)
        self.assertTensorEqual(
            module(EntityList.from_tensor_list(TensorList(
                torch.tensor([0, 1, 3, 6, 6]),
                torch.tensor([0, 2, 1, 0, 1, 0]),
            ))),
            torch.tensor([
                [1.0000, 1.0000, 1.0000],
                [1.1547, 1.1547, 1.1547],
                [1.0516, 1.0516, 1.0516],
                [0.0000, 0.0000, 0.0000],
            ]))

    def test_empty(self):
        embeddings = torch.empty((0, 3))
        module = FeaturizedEmbedding(weight=embeddings)
        self.assertTensorEqual(
            module(EntityList.from_tensor_list(TensorList(
                torch.zeros((1,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long)
            ))),
            torch.empty((0, 3)))

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


class TestIdentityOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = IdentityOperator(3)
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[0.3766, 0.9734, 0.5190],
                 [0.1801, 0.1585, 0.4585]],
                [[0.6188, 0.1917, 0.1006],
                 [0.3876, 0.7134, 0.7921]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())


class TestDiagonalOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = DiagonalOperator(3)
        with torch.no_grad():
            operator.diagonal += torch.arange(3, dtype=torch.float)
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[0.3766, 1.9468, 1.5570],
                 [0.1801, 0.3170, 1.3755]],
                [[0.6188, 0.3834, 0.3018],
                 [0.3876, 1.4268, 2.3763]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.diagonal.grad != 0).any())


class TestTranslationOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = TranslationOperator(3)
        with torch.no_grad():
            operator.translation += torch.arange(3, dtype=torch.float)
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[0.3766, 1.9734, 2.5190],
                 [0.1801, 1.1585, 2.4585]],
                [[0.6188, 1.1917, 2.1006],
                 [0.3876, 1.7134, 2.7921]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.translation.grad != 0).any())


class TestLinearOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = LinearOperator(3)
        with torch.no_grad():
            operator.linear_transformation += torch.arange(9, dtype=torch.float).view(3, 3)
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[ 2.3880,  8.5918, 13.7444],
                 [ 1.2556,  3.6253,  6.3166]],
                [[ 1.0117,  3.3179,  5.9601],
                 [ 2.6852,  8.6903, 14.4483]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.linear_transformation.grad != 0).any())


class TestAffineOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = AffineOperator(3)
        with torch.no_grad():
            operator.linear_transformation += torch.arange(9, dtype=torch.float).view(3, 3)
            operator.translation += torch.arange(3, dtype=torch.float)
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[ 2.3880,  9.5918, 15.7444],
                 [ 1.2556,  4.6253,  8.3166]],
                [[ 1.0117,  4.3179,  7.9601],
                 [ 2.6852,  9.6903, 16.4483]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.linear_transformation.grad != 0).any())
        self.assertTrue((operator.translation.grad != 0).any())


class TestComplexDiagonalOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190, 0.5453],
             [0.1801, 0.1585, 0.4585, 0.5928]],
            [[0.6188, 0.1917, 0.1006, 0.3378],
             [0.3876, 0.7134, 0.7921, 0.9434]],
        ], requires_grad=True)
        operator = ComplexDiagonalOperator(4)
        with torch.no_grad():
            operator.real[...] = torch.tensor([0.2949, 0.0029])
            operator.imag[...] = torch.tensor([0.4070, 0.1027])
        result = operator(embeddings)
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[-0.1002, -0.0532,  0.3063,  0.1015],
                 [-0.1335, -0.0604,  0.2085,  0.0180]],
                [[ 0.1415, -0.0341,  0.2815,  0.0207],
                 [-0.2081, -0.0948,  0.3913,  0.0760]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.real.grad != 0).any())
        self.assertTrue((operator.imag.grad != 0).any())


class TestIdentityDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = IdentityDynamicOperator(3, 5)
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[0.3766, 0.9734, 0.5190],
                 [0.1801, 0.1585, 0.4585]],
                [[0.6188, 0.1917, 0.1006],
                 [0.3876, 0.7134, 0.7921]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())


class TestDiagonalDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = DiagonalDynamicOperator(3, 5)
        with torch.no_grad():
            operator.diagonals += torch.arange(15, dtype=torch.float).view(5, 3)
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[0.3766, 1.9468, 1.5570],
                 [2.3413, 2.2190, 6.8775]],
                [[4.3316, 1.5336, 0.9054],
                 [0.3876, 1.4268, 2.3763]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.diagonals.grad != 0).any())


class TestTranslationDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = TranslationDynamicOperator(3, 5)
        with torch.no_grad():
            operator.translations += torch.arange(15, dtype=torch.float).view(5, 3)
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[ 0.3766,  1.9734,  2.5190],
                 [12.1801, 13.1585, 14.4585]],
                [[ 6.6188,  7.1917,  8.1006],
                 [ 0.3876,  1.7134,  2.7921]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.translations.grad != 0).any())


class TestLinearDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = LinearDynamicOperator(3, 5)
        with torch.no_grad():
            operator.linear_transformations += torch.arange(45, dtype=torch.float).view(5, 3, 3)
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[ 2.3880,  8.5918, 13.7444],
                 [29.9512, 32.3209, 35.0122]],
                [[17.4115, 19.7177, 22.3599],
                 [ 2.6852,  8.6903, 14.4483]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.linear_transformations.grad != 0).any())


class TestAffineDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190],
             [0.1801, 0.1585, 0.4585]],
            [[0.6188, 0.1917, 0.1006],
             [0.3876, 0.7134, 0.7921]],
        ], requires_grad=True)
        operator = AffineDynamicOperator(3, 5)
        with torch.no_grad():
            operator.linear_transformations += torch.arange(45, dtype=torch.float).view(5, 3, 3)
            operator.translations += torch.arange(15, dtype=torch.float).view(5, 3)
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[ 2.3880,  9.5918, 15.7444],
                 [41.9512, 45.3209, 49.0122]],
                [[23.4115, 26.7177, 30.3599],
                 [ 2.6852,  9.6903, 16.4483]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.linear_transformations.grad != 0).any())
        self.assertTrue((operator.translations.grad != 0).any())


class TestComplexDiagonalDynamicOperator(TensorTestCase):

    def test_forward(self):
        embeddings = torch.tensor([
            [[0.3766, 0.9734, 0.5190, 0.5453],
             [0.1801, 0.1585, 0.4585, 0.5928]],
            [[0.6188, 0.1917, 0.1006, 0.3378],
             [0.3876, 0.7134, 0.7921, 0.9434]],
        ], requires_grad=True)
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
        result = operator(embeddings, torch.tensor([[0, 4], [2, 0]]))
        self.assertTensorEqual(
            result,
            torch.tensor([
                [[-0.1002, -0.0532,  0.3063,  0.1015],
                 [-0.3845,  0.0201,  0.3521,  0.5935]],
                [[ 0.1582,  0.1196,  0.3376,  0.3627],
                 [-0.2081, -0.0948,  0.3913,  0.0760]],
            ]))
        result.sum().backward()
        self.assertTrue((embeddings.grad != 0).any())
        self.assertTrue((operator.real.grad != 0).any())
        self.assertTrue((operator.imag.grad != 0).any())


class TestDotComparator(TensorTestCase):

    def test_forward_one_batch(self):
        comparator = DotComparator()
        lhs_pos = torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True)
        rhs_pos = torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True)
        lhs_neg = torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.0000, 0.0000, 0.0000],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True)
        rhs_neg = torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.0000, 0.0000, 0.0000],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[1.2024, 0.3246]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[0.6463, 1.4433, 0.0000, 1.1491],
                 [0.5392, 0.7652, 0.0000, 0.5815]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[1.0838, 0.0000, 0.6631, 0.3002],
                 [0.9457, 0.0000, 0.6156, 0.2751]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())

    def test_forward_two_batches(self):
        comparator = DotComparator()
        lhs_pos = torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True)
        rhs_pos = torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True)
        lhs_neg = torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.0000, 0.0000, 0.0000],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True)
        rhs_neg = torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.0000, 0.0000, 0.0000]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[1.2024], [0.3246]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[0.6463, 1.4433]],
                [[0.0000, 0.5815]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[1.0838, 0.0000]],
                [[0.6156, 0.2751]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())


class TestCosComparator(TensorTestCase):

    def test_forward_one_batch(self):
        comparator = CosComparator()
        lhs_pos = torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True)
        rhs_pos = torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True)
        lhs_neg = torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True)
        rhs_neg = torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[0.9741, 0.6106]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[0.6165, 0.8749, 0.9664, 0.8701],
                 [0.9607, 0.8663, 0.7494, 0.8224]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[0.8354, 0.6406, 0.6626, 0.6856],
                 [0.9063, 0.7439, 0.7648, 0.7810]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())

    def test_forward_two_batches(self):
        comparator = CosComparator()
        lhs_pos = torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True)
        rhs_pos = torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True)
        lhs_neg = torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True)
        rhs_neg = torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[0.9741], [0.6106]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[0.6165, 0.8749]],
                [[0.7494, 0.8224]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[0.8354, 0.6406]],
                [[0.7648, 0.7810]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())


class TestL2Comparator(TensorTestCase):

    def test_forward_one_batch(self):
        comparator = L2Comparator()
        lhs_pos = torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True)
        rhs_pos = torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True)
        lhs_neg = torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True)
        rhs_neg = torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[-0.3246, -0.6639]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[-0.9650, -0.6569, -0.5992, -0.6006],
                 [-0.2965, -0.8546, -0.4666, -0.6621]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[-0.7056, -0.9015, -0.8221, -0.7835],
                 [-0.6412, -0.7378, -0.6388, -0.5489]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())

    def test_forward_two_batches(self):
        comparator = L2Comparator()
        lhs_pos = torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True)
        rhs_pos = torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True)
        lhs_neg = torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True)
        rhs_neg = torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[-0.3246], [-0.6639]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[-0.9650, -0.6569]],
                [[-0.4666, -0.6621]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[-0.7056, -0.9015]],
                [[-0.6388, -0.5489]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())


class TestSquaredL2Comparator(TensorTestCase):

    def test_forward_one_batch(self):
        comparator = SquaredL2Comparator()
        lhs_pos = torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True)
        rhs_pos = torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True)
        lhs_neg = torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True)
        rhs_neg = torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[-0.1054, -0.4407]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[-0.9312, -0.4315, -0.3591, -0.3608],
                 [-0.0879, -0.7303, -0.2177, -0.4384]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[-0.4979, -0.8127, -0.6759, -0.6138],
                 [-0.4112, -0.5443, -0.4080, -0.3013]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())

    def test_forward_two_batches(self):
        comparator = SquaredL2Comparator()
        lhs_pos = torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True)
        rhs_pos = torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True)
        lhs_neg = torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True)
        rhs_neg = torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[-0.1054], [-0.4407]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[-0.9312, -0.4315]],
                [[-0.2177, -0.4384]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[-0.4979, -0.8127]],
                [[-0.4080, -0.3013]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())


class TestBiasedComparator(TensorTestCase):

    def test_forward_one_batch(self):
        comparator = BiasedComparator(CosComparator())
        lhs_pos = torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True)
        rhs_pos = torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True)
        lhs_neg = torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True)
        rhs_neg = torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[2.8086, 1.5434]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[1.7830, 2.5800, 2.3283, 2.4269],
                 [1.5172, 2.0194, 1.4850, 1.9369]]
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[2.5017, 2.0980, 2.1129, 1.9578],
                 [2.2670, 1.8759, 1.8838, 1.7381]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())

    def test_forward_two_batches(self):
        comparator = BiasedComparator(CosComparator())
        lhs_pos = torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True)
        rhs_pos = torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True)
        lhs_neg = torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True)
        rhs_neg = torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True)
        pos_scores, lhs_neg_scores, rhs_neg_scores = comparator(
            comparator.prepare(lhs_pos), comparator.prepare(rhs_pos),
            comparator.prepare(lhs_neg), comparator.prepare(rhs_neg))
        self.assertTensorEqual(pos_scores, torch.tensor([[2.8086], [1.5434]]))
        self.assertTensorEqual(
            lhs_neg_scores,
            torch.tensor([
                [[1.7830, 2.5800]],
                [[1.4850, 1.9369]],
            ]))
        self.assertTensorEqual(
            rhs_neg_scores,
            torch.tensor([
                [[2.5017, 2.0980]],
                [[1.8838, 1.7381]],
            ]))
        (pos_scores.sum() + lhs_neg_scores.sum() + rhs_neg_scores.sum()).backward()
        self.assertTrue((lhs_pos.grad != 0).any())
        self.assertTrue((rhs_pos.grad != 0).any())
        self.assertTrue((lhs_neg.grad != 0).any())
        self.assertTrue((rhs_neg.grad != 0).any())


if __name__ == '__main__':
    main()
