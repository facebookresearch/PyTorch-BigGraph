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

from enum import Enum
from unittest import TestCase

import torch
import torch.nn as nn
from tensorlist.tensorlist import TensorList

from torchbiggraph.config import Operator, Metric, LossFn, EntitySchema, \
    RelationSchema
from torchbiggraph.model import (
    match_shape,
    # Embeddings
    SimpleEmbedding, FeaturizedEmbedding,
    # Operators
    IdentityOperator, DiagonalOperator, TranslationOperator, AffineOperator,
    # Dynamic operators
    DiagonalDynamicOperator, TranslationDynamicOperator,
    # Metric
    DotMetric, CosMetric, BiasedMetric,
    # Losses
    LogisticLoss, RankingLoss, SoftmaxLoss,
    # Model
    MultiRelationEmbedder,
)
from torchbiggraph.util import Side


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
            operator.rotation += torch.arange(9, dtype=torch.float).view(3, 3)
            operator.translation += torch.arange(3, dtype=torch.float)
        assertTensorEqual(self, operator(embeddings), torch.tensor([
            [[ 6.4108,  9.8766, 12.2912],
             [ 3.4066,  5.1821,  7.2792]],
            [[ 1.7975,  3.2815,  5.1015],
             [ 7.2804, 10.4993, 13.4711]],
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


class TestDotMetric(TestCase):

    def test_forward_one_batch(self):
        metric = DotMetric()
        lhs_pos = metric.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.0000, 0.0000, 0.0000],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.0000, 0.0000, 0.0000],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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
        metric = DotMetric()
        lhs_pos = metric.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.0000, 0.0000, 0.0000],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.0000, 0.0000, 0.0000]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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


class TestCosMetric(TestCase):

    def test_forward_one_batch(self):
        metric = CosMetric()
        lhs_pos = metric.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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
        metric = CosMetric()
        lhs_pos = metric.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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


class TestBiasedMetric(TestCase):

    def test_forward_one_batch(self):
        metric = BiasedMetric(CosMetric())
        lhs_pos = metric.prepare(torch.tensor([[
            [0.8931, 0.2241, 0.4241],
            [0.6557, 0.2492, 0.4157],
        ]], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([[
            [0.9220, 0.2892, 0.7408],
            [0.1476, 0.6079, 0.1835],
        ]], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([[
            [0.3836, 0.7648, 0.0965],
            [0.8929, 0.8947, 0.4877],
            [0.4754, 0.3163, 0.3422],
            [0.7967, 0.6736, 0.2966],
        ]], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([[
            [0.6116, 0.6010, 0.9500],
            [0.2541, 0.7715, 0.7477],
            [0.2360, 0.5923, 0.7536],
            [0.1290, 0.3088, 0.2731],
        ]], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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
        metric = BiasedMetric(CosMetric())
        lhs_pos = metric.prepare(torch.tensor([
            [[0.8931, 0.2241, 0.4241]],
            [[0.6557, 0.2492, 0.4157]],
        ], requires_grad=True))
        rhs_pos = metric.prepare(torch.tensor([
            [[0.9220, 0.2892, 0.7408]],
            [[0.1476, 0.6079, 0.1835]],
        ], requires_grad=True))
        lhs_neg = metric.prepare(torch.tensor([
            [[0.3836, 0.7648, 0.0965],
             [0.8929, 0.8947, 0.4877]],
            [[0.4754, 0.3163, 0.3422],
             [0.7967, 0.6736, 0.2966]],
        ], requires_grad=True))
        rhs_neg = metric.prepare(torch.tensor([
            [[0.6116, 0.6010, 0.9500],
             [0.2541, 0.7715, 0.7477]],
            [[0.2360, 0.5923, 0.7536],
             [0.1290, 0.3088, 0.2731]],
        ], requires_grad=True))
        pos_scores, lhs_neg_scores, rhs_neg_scores = \
            metric(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
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


class Entities(Enum):
    SIMPLE = "simple"
    FEATURIZED = "featurized"


class Relations(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class Negatives(Enum):
    SAME_BATCH = "same_batch"
    SAME_BATCH_PLUS_UNIFORM = "same_batch_plus_uniform"
    ALL = "all"


def modify_operator(operator, delta):
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter += delta


class TestModel(TestCase):

    def make_model(
        self,
        entities: Entities,
        relations: Relations,
        negatives: Negatives,
        operator: Operator,
        metric: Metric,
        loss_fn: LossFn,
        *,
        bias: bool = False,
    ) -> MultiRelationEmbedder:
        torch.manual_seed(42)
        # 10 embeddings in 5 dimensions, normalized to be on the unit sphere.
        lhs_emb = nn.Parameter(torch.tensor([
            [ 0.1849, -0.2634,  0.4323,  0.3014, -0.7866],
            [-0.1792, -0.2467,  0.4516,  0.6828, -0.4867],
            [ 0.5515,  0.1612, -0.4522,  0.1214, -0.6713],
            [-0.2804, -0.4862,  0.5401, -0.6045, -0.1670],
            [-0.4249, -0.4165,  0.7658,  0.2435, -0.0163],
            [-0.2746, -0.2862,  0.5354, -0.3321, -0.6677],
            [-0.4629, -0.5577,  0.6125, -0.1057,  0.2972],
            [-0.0127, -0.3991, -0.0148,  0.5342, -0.7450],
            [-0.8753, -0.1322, -0.0396,  0.4534,  0.0958],
            [ 0.6884,  0.4331,  0.3418,  0.4403,  0.1669],
        ]))
        # Another 10.
        rhs_emb = nn.Parameter(torch.tensor([
            [-0.3246, -0.2619,  0.0344, -0.7948, -0.4394],
            [ 0.4379,  0.1108,  0.0133,  0.5760,  0.6812],
            [ 0.3697, -0.0362, -0.7794,  0.0266, -0.5038],
            [-0.2528, -0.3999, -0.0681,  0.6444,  0.5970],
            [ 0.0944,  0.3583, -0.2588, -0.7607,  0.4659],
            [ 0.4479, -0.3611,  0.5052,  0.6258,  0.1489],
            [-0.3459, -0.6755, -0.6424,  0.0733,  0.0772],
            [-0.2570,  0.5254,  0.7824,  0.0317, -0.2116],
            [ 0.4466,  0.8831,  0.1014, -0.0679,  0.0763],
            [-0.2768,  0.1322, -0.7094, -0.2896,  0.5646],
        ]))
        entity_dict = {"foo_entity": EntitySchema(
            numPartitions=1,
            featurized=1 if entities is Entities.FEATURIZED else 0,
        )}
        relation_list = [RelationSchema(
            name="bar_relation",
            lhs="foo_entity",
            rhs="foo_entity",
            weight=1.0,
            operator=operator,
            all_rhs_negs=1 if negatives is Negatives.ALL else 0,
        )]
        model = MultiRelationEmbedder(
            dim=5,
            relations=relation_list,
            entities=entity_dict,
            num_batch_negs=2,
            num_uniform_negs=2 if negatives is Negatives.SAME_BATCH_PLUS_UNIFORM else 0,
            margin=0.1,
            metric=metric,
            global_emb=False,
            max_norm=None,
            loss_fn=loss_fn,
            bias=bias,
            num_dynamic_rels=1 if relations is Relations.DYNAMIC else 0,
        )
        if relations is Relations.STATIC:
            modify_operator(model.rhs_operators[0], torch.arange(5, dtype=torch.float))
        elif relations is Relations.DYNAMIC:
            modify_operator(model.lhs_operators[0], torch.arange(5, dtype=torch.float))
            modify_operator(model.rhs_operators[0], -torch.arange(5, dtype=torch.float))
        model.set_embeddings("foo_entity", lhs_emb, Side.LHS)
        model.set_embeddings("foo_entity", rhs_emb, Side.RHS)
        return model

    def test_static_relations_same_batch_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.2988))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00],
            [-5.5695e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01],
            [ 1.1259e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_static_relations_same_batch_plus_uniform_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH_PLUS_UNIFORM,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(7.3405))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00,  1.1728e+00,  6.9172e-01],
            [-5.5695e-01, -1.0000e+09,  1.0000e-01, -1.5759e-02],
            [-1.0000e+09, -1.0000e+09,  8.8518e-01,  4.8799e-01],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01,  1.4938e-01,  2.2449e-01],
            [ 1.1259e+00, -1.0000e+09,  1.8193e-01,  1.7039e-01],
            [-1.0000e+09, -1.0000e+09,  8.7782e-01,  1.0000e-01],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01,  8.9193e-01,  4.1083e-01],
            [-7.9096e-01, -1.0000e+09, -1.3401e-01, -2.4977e-01],
            [-1.0000e+09, -1.0000e+09, -1.8317e-01, -5.8036e-01],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01, -1.3152e-01, -5.6411e-02],
            [ 8.9193e-01, -1.0000e+09, -5.2087e-02, -6.3628e-02],
            [-1.0000e+09, -1.0000e+09, -1.9052e-01, -9.6834e-01],
        ]))
        loss.backward()

    def test_static_relations_all_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.ALL,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(13.7306))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [ 1.0207e+00, -5.1007e-01, -1.0000e+09, -1.8432e-01, -4.4586e-02,  2.2449e-01,  1.4938e-01,  7.5075e-01, -6.8589e-02, -3.4155e-01],
            [ 1.9570e-01, -1.0000e+09,  1.1259e+00, -2.6161e-01,  5.5756e-02,  1.7039e-01,  1.8193e-01, -3.0932e-02,  5.1735e-01,  9.2873e-03],
            [ 8.7782e-01,  1.0303e+00,  9.2241e-01,  1.3310e+00, -1.0000e+09,  1.6601e+00,  1.0193e+00,  1.4627e+00,  7.3275e-01,  2.9244e-01],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [ 1.0207e+00, -5.1007e-01, -1.0000e+09, -1.8432e-01, -4.4586e-02,  2.2449e-01,  1.4938e-01,  7.5075e-01, -6.8589e-02, -3.4155e-01],
            [ 1.9570e-01, -1.0000e+09,  1.1259e+00, -2.6161e-01,  5.5756e-02,  1.7039e-01,  1.8193e-01, -3.0932e-02,  5.1735e-01,  9.2873e-03],
            [ 8.7782e-01,  1.0303e+00,  9.2241e-01,  1.3310e+00, -1.0000e+09,  1.6601e+00,  1.0193e+00,  1.4627e+00,  7.3275e-01,  2.9244e-01],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [ 7.3985e-01, -7.9096e-01, -1.0000e+09, -4.6521e-01, -3.2548e-01, -5.6411e-02, -1.3152e-01,  4.6986e-01, -3.4949e-01, -6.2245e-01],
            [-3.8310e-02, -1.0000e+09,  8.9193e-01, -4.9562e-01, -1.7826e-01, -6.3628e-02, -5.2087e-02, -2.6495e-01,  2.8334e-01, -2.2473e-01],
            [-1.9052e-01, -3.8047e-02, -1.4593e-01,  2.6264e-01, -1.0000e+09,  5.9179e-01, -4.9001e-02,  3.9440e-01, -3.3560e-01, -7.7591e-01],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [ 7.3985e-01, -7.9096e-01, -1.0000e+09, -4.6521e-01, -3.2548e-01, -5.6411e-02, -1.3152e-01,  4.6986e-01, -3.4949e-01, -6.2245e-01],
            [-3.8310e-02, -1.0000e+09,  8.9193e-01, -4.9562e-01, -1.7826e-01, -6.3628e-02, -5.2087e-02, -2.6495e-01,  2.8334e-01, -2.2473e-01],
            [-1.9052e-01, -3.8047e-02, -1.4593e-01,  2.6264e-01, -1.0000e+09,  5.9179e-01, -4.9001e-02,  3.9440e-01, -3.3560e-01, -7.7591e-01],
        ]))
        loss.backward()

    def test_dynamic_relations_same_batch_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.DYNAMIC,
            Negatives.SAME_BATCH,
            Operator.TRANSLATION,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            torch.tensor([0, 0, 0]),
        )
        assertTensorEqual(self, loss, torch.tensor(8.9651))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.3545e+00],
            [-7.3865e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09,  7.6105e+00],
            [-6.9947e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [2.7016],
            [2.9302],
            [-1.7264],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-3.7113],
            [4.4562],
            [-1.5461],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.9561e+00],
            [ 2.0915e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.7992e+00],
            [-2.6385e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_dynamic_relations_same_batch_plus_uniform_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.DYNAMIC,
            Negatives.SAME_BATCH_PLUS_UNIFORM,
            Operator.TRANSLATION,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            torch.tensor([0, 0, 0]),
        )
        assertTensorEqual(self, loss, torch.tensor(19.7492))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.3545e+00,  1.3545e+00, -3.8468e-01],
            [-7.3865e-01, -1.0000e+09,  1.0000e-01, -1.2739e+00],
            [-1.0000e+09, -1.0000e+09,  1.0428e-01, -5.3431e-01],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09,  7.6105e+00,  2.2482e+00,  6.8772e+00],
            [-6.9947e+00, -1.0000e+09, -5.8399e+00, -1.2975e+00],
            [-1.0000e+09, -1.0000e+09, -2.8795e+00,  1.0000e-01],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [2.7016],
            [2.9302],
            [-1.7264],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-3.7113],
            [4.4562],
            [-1.5461],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.9561e+00,  3.9561e+00,  2.2169e+00],
            [ 2.0915e+00, -1.0000e+09,  2.9302e+00,  1.5563e+00],
            [-1.0000e+09, -1.0000e+09, -1.7222e+00, -2.3608e+00],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.7992e+00, -1.5631e+00,  3.0659e+00],
            [-2.6385e+00, -1.0000e+09, -1.4837e+00,  3.0587e+00],
            [-1.0000e+09, -1.0000e+09, -4.5256e+00, -1.5461e+00],
        ]))
        loss.backward()

    def test_dynamic_relations_all_negatives(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.DYNAMIC,
            Negatives.ALL,
            Operator.TRANSLATION,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            torch.tensor([0, 0, 0]),
        )
        assertTensorEqual(self, loss, torch.tensor(84.5526))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-8.1534e-01, -3.5056e+00,  1.3545e+00, -1.1531e+00, -5.1062e+00, -1.0000e+09, -4.9215e+00, -3.8468e-01, -4.4578e+00, -5.8068e+00],
            [-1.4939e+00, -3.6263e+00, -1.0000e+09, -1.5741e+00, -4.7035e+00, -7.3865e-01, -4.4840e+00, -1.2739e+00, -4.4342e+00, -5.2141e+00],
            [ 2.6829e+00, -1.0000e+09,  4.7124e+00,  3.7555e+00, -5.3431e-01,  4.3835e+00,  1.0428e-01,  2.7387e+00, -1.2557e-01, -1.4042e+00],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [ 2.1605e-01,  7.6105e+00, -1.0000e+09,  7.1312e+00,  2.9080e+00,  6.8772e+00,  2.2482e+00,  5.6201e+00,  4.6492e+00,  3.2918e+00],
            [-8.7296e+00, -1.0000e+09, -6.9947e+00, -1.0667e+00, -5.1122e+00, -1.2975e+00, -5.8399e+00, -3.2822e+00, -2.8854e+00, -4.4779e+00],
            [-2.8795e+00,  6.1983e+00, -2.0302e+00,  5.6939e+00, -1.0000e+09,  5.3602e+00,  1.6554e-01,  3.3794e+00,  2.4979e+00,  9.7324e-01],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [2.7016],
            [2.9302],
            [-1.7264],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-3.7113],
            [4.4562],
            [-1.5461],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [ 1.7863e+00, -9.0403e-01,  3.9561e+00,  1.4485e+00, -2.5046e+00, -1.0000e+09, -2.3199e+00,  2.2169e+00, -1.8562e+00, -3.2051e+00],
            [ 1.3363e+00, -7.9615e-01, -1.0000e+09,  1.2561e+00, -1.8733e+00,  2.0915e+00, -1.6538e+00,  1.5563e+00, -1.6041e+00, -2.3839e+00],
            [ 8.5645e-01, -1.0000e+09,  2.8859e+00,  1.9291e+00, -2.3608e+00,  2.5570e+00, -1.7222e+00,  9.1227e-01, -1.9520e+00, -3.2307e+00],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-3.5953e+00,  3.7992e+00, -1.0000e+09,  3.3199e+00, -9.0328e-01,  3.0659e+00, -1.5631e+00,  1.8088e+00,  8.3791e-01, -5.1945e-01],
            [-4.3734e+00, -1.0000e+09, -2.6385e+00,  3.2895e+00, -7.5606e-01,  3.0587e+00, -1.4837e+00,  1.0740e+00,  1.4707e+00, -1.2173e-01],
            [-4.5256e+00,  4.5522e+00, -3.6763e+00,  4.0477e+00, -1.0000e+09,  3.7141e+00, -1.4806e+00,  1.7333e+00,  8.5180e-01, -6.7291e-01],
        ]))
        loss.backward()

    def test_static_relations_none_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.2988))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00],
            [-5.5695e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01],
            [ 1.1259e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_static_relations_diagonal_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.DIAGONAL,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(7.5406))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  2.7395e+00],
            [-1.3541e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -3.4156e+00],
            [ 4.8010e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [0.3139],
            [-1.7476],
            [-3.7557],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [0.3139],
            [-1.7476],
            [-3.7557],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  2.9535e+00],
            [-3.2017e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -3.2017e+00],
            [ 2.9535e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_static_relations_translation_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.TRANSLATION,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.1171))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  9.9113e-01],
            [-3.7525e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01],
            [ 1.1259e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-3.0634],
            [-3.1982],
            [-0.2102],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-3.0634],
            [-3.1982],
            [-0.2102],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09, -2.1723e+00],
            [-3.6735e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -3.6735e+00],
            [-2.1723e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_static_relations_affine_rhs_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.AFFINE_RHS,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(10.6878))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1589e+00],
            [-4.4701e-02, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -8.4147e+00],
            [ 9.5289e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.4026],
            [-8.7726],
            [-0.2867],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.4026],
            [-8.7726],
            [-0.2867],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  6.5629e-01],
            [-8.9173e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -8.9173e+00],
            [ 6.5629e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_dynamic_relations_diagonal_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.DYNAMIC,
            Negatives.SAME_BATCH,
            Operator.DIAGONAL,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            torch.tensor([0, 0, 0]),
        )
        assertTensorEqual(self, loss, torch.tensor(5.0412))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09, -3.9389e-01],
            [ 2.4019e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -3.4156e+00],
            [ 4.8010e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.6757],
            [1.4795],
            [1.8190],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [0.3139],
            [-1.7476],
            [-3.7557],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09, -1.1696e+00],
            [ 1.6197e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -3.2017e+00],
            [ 2.9535e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_dynamic_relations_translation_operator(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.DYNAMIC,
            Negatives.SAME_BATCH,
            Operator.TRANSLATION,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            torch.tensor([0, 0, 0]),
        )
        assertTensorEqual(self, loss, torch.tensor(8.9651))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.3545e+00],
            [-7.3865e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09,  7.6105e+00],
            [-6.9947e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [2.7016],
            [2.9302],
            [-1.7264],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-3.7113],
            [4.4562],
            [-1.5461],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.9561e+00],
            [ 2.0915e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09,  3.7992e+00],
            [-2.6385e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_bias(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=True,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.7889))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.6935e+00],
            [-1.0213e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -4.2314e-01],
            [ 1.0954e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [0.0157],
            [0.6139],
            [-1.0362],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [0.0157],
            [0.6139],
            [-1.0362],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  1.6092e+00],
            [-5.0742e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -5.0742e-01],
            [ 1.6092e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_dot(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.2988))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00],
            [-5.5695e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01],
            [ 1.1259e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_cos(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.COS,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.2988))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00],
            [-5.5691e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1003e-01],
            [ 1.1260e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9194e-01],
            [-7.9092e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9092e-01],
            [ 8.9194e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_ranking_loss(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.RANKING,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(2.2988))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [-1.0000e+09,  1.1728e+00],
            [-5.5695e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [-1.0000e+09, -5.1007e-01],
            [ 1.1259e+00, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_logistic_loss(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.LOGISTIC,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(7.2899))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()

    def test_softmax_loss(self):
        model = self.make_model(
            Entities.SIMPLE,
            Relations.STATIC,
            Negatives.SAME_BATCH,
            Operator.NONE,
            Metric.DOT,
            LossFn.SOFTMAX,
            bias=False,
        )
        loss, margins, scores = model(
            torch.tensor([5, 2, 1]),
            torch.tensor([2, 1, 4]),
            0,
        )
        assertTensorEqual(self, loss, torch.tensor(3.5509))
        lhs_margin, rhs_margin = margins
        assertTensorEqual(self, lhs_margin, torch.tensor([
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ]))
        assertTensorEqual(self, rhs_margin, torch.tensor([
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ]))
        lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores = scores
        assertTensorEqual(self, lhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, rhs_pos_scores, torch.tensor([
            [-0.1809],
            [-0.1340],
            [-0.9683],
        ]))
        assertTensorEqual(self, lhs_neg_scores, torch.tensor([
            [-1.0000e+09,  8.9193e-01],
            [-7.9096e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        assertTensorEqual(self, rhs_neg_scores, torch.tensor([
            [-1.0000e+09, -7.9096e-01],
            [ 8.9193e-01, -1.0000e+09],
            [-1.0000e+09, -1.0000e+09],
        ]))
        loss.backward()
