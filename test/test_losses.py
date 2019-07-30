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

from torchbiggraph.losses import LogisticLossFunction, RankingLossFunction, SoftmaxLossFunction


class TensorTestCase(TestCase):

    def assertTensorEqual(self, actual, expected):
        if not isinstance(actual, torch.FloatTensor):
            self.fail("Expected FloatTensor, got %s" % type(actual))
        if actual.size() != expected.size():
            self.fail("Expected tensor of size %s, got %s"
                      % (expected.size(), actual.size()))
        if not torch.allclose(actual, expected, rtol=0.00005, atol=0.00005, equal_nan=True):
            self.fail("Expected\n%r\ngot\n%r" % (expected, actual))


class TestLogisticLossFunction(TensorTestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = LogisticLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(4.2589))
        loss.backward()
        self.assertTrue((pos_scores.grad != 0).any())
        self.assertTrue((neg_scores.grad != 0).any())

    def test_forward_good(self):
        pos_scores = torch.full((3,), +1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), -1e9, requires_grad=True)
        loss_fn = LogisticLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), +1e9, requires_grad=True)
        loss_fn = LogisticLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(6e9))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.zeros((3,), requires_grad=True)
        neg_scores = torch.empty((3, 0), requires_grad=True)
        loss_fn = LogisticLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(2.0794))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.empty((0,), requires_grad=True)
        neg_scores = torch.empty((0, 0), requires_grad=True)
        loss_fn = LogisticLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()


class TestRankingLossFunction(TensorTestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = RankingLossFunction(1.)
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(13.4475))
        loss.backward()
        self.assertTrue((pos_scores.grad != 0).any())
        self.assertTrue((neg_scores.grad != 0).any())

    def test_forward_good(self):
        pos_scores = torch.full((3,), 2, requires_grad=True)
        neg_scores = torch.full((3, 5), 1, requires_grad=True)
        loss_fn = RankingLossFunction(1.)
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1, requires_grad=True)
        neg_scores = torch.zeros((3, 5), requires_grad=True)
        loss_fn = RankingLossFunction(1.)
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(30.))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.zeros((3,), requires_grad=True)
        neg_scores = torch.empty((3, 0), requires_grad=True)
        loss_fn = RankingLossFunction(1.)
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.empty((0,), requires_grad=True)
        neg_scores = torch.empty((0, 3), requires_grad=True)
        loss_fn = RankingLossFunction(1.)
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()


class TestSoftmaxLossFunction(TensorTestCase):

    def test_forward(self):
        pos_scores = torch.tensor([0.8181, 0.5700, 0.3506], requires_grad=True)
        neg_scores = torch.tensor([
            [0.4437, 0.6573, 0.9986, 0.2548, 0.0998],
            [0.6175, 0.4061, 0.4582, 0.5382, 0.3126],
            [0.9869, 0.2028, 0.1667, 0.0044, 0.9934],
        ], requires_grad=True)
        loss_fn = SoftmaxLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(5.2513))
        loss.backward()
        self.assertTrue((pos_scores.grad != 0).any())
        self.assertTrue((neg_scores.grad != 0).any())

    def test_forward_good(self):
        pos_scores = torch.full((3,), +1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), -1e9, requires_grad=True)
        loss_fn = SoftmaxLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()

    def test_forward_bad(self):
        pos_scores = torch.full((3,), -1e9, requires_grad=True)
        neg_scores = torch.full((3, 5), +1e9, requires_grad=True)
        loss_fn = SoftmaxLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.tensor(6e9))
        loss.backward()

    def test_no_neg(self):
        pos_scores = torch.zeros((3,), requires_grad=True)
        neg_scores = torch.empty((3, 0), requires_grad=True)
        loss_fn = SoftmaxLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()

    def test_no_pos(self):
        pos_scores = torch.empty((0,), requires_grad=True)
        neg_scores = torch.empty((0, 3), requires_grad=True)
        loss_fn = SoftmaxLossFunction()
        loss = loss_fn(pos_scores, neg_scores)
        self.assertTensorEqual(loss, torch.zeros(()))
        loss.backward()


if __name__ == '__main__':
    main()
