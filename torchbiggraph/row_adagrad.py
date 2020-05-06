#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging

from torch.optim import Optimizer


logger = logging.getLogger("torchbiggraph")


class RowAdagrad(Optimizer):
    """Implements a row-wise variant of the Adagrad algorithm.
    Assumes that all the model parameters are 2-dimensional tensors
    containing embedding weights.

    Code mostly copy-pasted from torch/optim/Adagrad, with HOGWILD-safe
    update (see async_adagrad.py)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0):
        # lr_decay is a little tricky beause keeping track of # of steps
        # is not straightforward when they're happening in a distributed way.
        # Anyway, we don't use lr_decay in Filament anyway
        assert lr_decay == 0, "lr_decay not currently supported."
        defaults = {"lr": lr, "lr_decay": lr_decay, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                # state['step'] = 0
                if p.dim() != 2:
                    raise ValueError("RowAdagrad only works on 2D tensors")
                state = self.state[p]
                state["sum"] = p.new_zeros((p.shape[0],))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # state['step'] += 1

                if group["weight_decay"] != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not "
                            "compatible with sparse gradients "
                        )
                    grad = grad.add(group["weight_decay"], p.data)

                # clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                clr = group["lr"]

                if grad.is_sparse:
                    if grad._indices().numel() == 0:
                        continue
                    # the update is non-linear so indices must be unique
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    # multiple HOGWILD processes may perform unsynchronized
                    # updates to G. Update a local copy of G independently from
                    # the shared-memory copy, to guarantee that
                    # local_G >= grad^2
                    local_G = state["sum"][grad_indices]  # _sparse_mask
                    delta_G = (grad_values * grad_values).mean(1)
                    state["sum"].index_add_(0, grad_indices, delta_G)
                    local_G += delta_G
                    std_values = local_G.sqrt_().add_(1e-10).unsqueeze(1)
                    p.data.index_add_(0, grad_indices, -clr * grad_values / std_values)
                else:
                    # multiple HOGWILD processes may perform unsynchronized
                    # updates to G. Update a local copy of G independently from
                    # the shared-memory copy, to guarantee that
                    # local_G >= grad^2
                    local_G = state["sum"].clone()
                    delta_G = (grad * grad).mean(1)
                    state["sum"] += delta_G
                    local_G += delta_G
                    std = local_G.sqrt().add_(1e-10)
                    p.data.addcdiv_(grad, std.unsqueeze(1), value=-clr)

        return loss
