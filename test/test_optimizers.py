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

import logging
import os
import unittest
from unittest import main, TestCase

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adagrad
from torchbiggraph.async_adagrad import AsyncAdagrad
from torchbiggraph.row_adagrad import RowAdagrad


logger = logging.getLogger("torchbiggraph")


class TensorTestCase(TestCase):
    def assertTensorEqual(self, actual, expected):
        if not isinstance(actual, (torch.FloatTensor, torch.cuda.FloatTensor)):
            self.fail("Expected FloatTensor, got %s" % type(actual))
        if actual.size() != expected.size():
            self.fail(
                "Expected tensor of size %s, got %s" % (expected.size(), actual.size())
            )
        if not torch.allclose(
            actual, expected, rtol=0.00005, atol=0.00005, equal_nan=True
        ):
            self.fail("Expected\n%r\ngot\n%r" % (expected, actual))


def do_optim(model, optimizer, N, rank):
    torch.random.manual_seed(rank)
    for i in range(N):
        optimizer.zero_grad()
        NE = model.weight.shape[0]
        inputs = (torch.rand(10) * NE).long()
        L = model(inputs).sum()
        L.backward()
        # print(next(model.parameters()).grad)
        optimizer.step()


class TestOptimizers(TensorTestCase):
    def _stress_optimizer(self, model, optimizer, num_processes=1, iterations=100):
        logger.info("_stress_optimizer begin")
        processes = []
        for rank in range(num_processes):
            p = mp.get_context("spawn").Process(
                name=f"Process-{rank}",
                target=do_optim,
                args=(model, optimizer, iterations, rank),
            )
            p.start()
            self.addCleanup(p.terminate)
            processes.append(p)

        for p in processes:
            p.join()

        logger.info("_stress_optimizer complete")

    # def testHogwildStability_Adagrad(self):
    #     NE = 10000
    #     model = nn.Embedding(NE, 100)
    #     optimizer = Adagrad(model.parameters())
    #     num_processes = mp.cpu_count() // 2 + 1
    #     self._stress_optimizer(model, optimizer, num_processes)

    #     # This fails for Adagrad because it's not stable
    #     self.assertLess(model.weight.abs().max(), 1000)

    @unittest.skipIf(os.environ.get("CIRCLECI_TEST") == "1", "Hangs in CircleCI")
    def testHogwildStability_AsyncAdagrad(self):
        NE = 10000
        model = nn.Embedding(NE, 100)
        optimizer = AsyncAdagrad(model.parameters())
        num_processes = mp.cpu_count() // 2 + 1
        self._stress_optimizer(
            model, optimizer, num_processes=num_processes, iterations=50
        )

        self.assertLess(model.weight.abs().max(), 1000)

    @unittest.skipIf(os.environ.get("CIRCLECI_TEST") == "1", "Hangs in CircleCI")
    def testHogwildStability_RowAdagrad(self):
        NE = 10000
        model = nn.Embedding(NE, 100)
        optimizer = RowAdagrad(model.parameters())
        num_processes = mp.cpu_count() // 2 + 1
        self._stress_optimizer(
            model, optimizer, num_processes=num_processes, iterations=50
        )

        # This fails for Adagrad because it's not stable
        self.assertLess(model.weight.abs().max(), 1000)

    def _assert_testAccuracy_AsyncAdagrad(self, sparse):
        # testing that Adagrad = AsyncAdagrad with 1 process
        NE = 10000
        golden_model = nn.Embedding(NE, 100, sparse=sparse)
        test_model = nn.Embedding(NE, 100, sparse=sparse)
        test_model.load_state_dict(golden_model.state_dict())

        golden_optimizer = Adagrad(golden_model.parameters())
        self._stress_optimizer(golden_model, golden_optimizer, num_processes=1)

        test_optimizer = AsyncAdagrad(test_model.parameters())
        self._stress_optimizer(test_model, test_optimizer, num_processes=1)

        # This fails for Adagrad because it's not stable
        self.assertTensorEqual(golden_model.weight, test_model.weight)

    def testAccuracy_AsyncAdagrad_sprase_true(self):
        self._assert_testAccuracy_AsyncAdagrad(sparse=True)

    def testAccuracy_AsyncAdagrad_sprase_false(self):
        self._assert_testAccuracy_AsyncAdagrad(sparse=False)
