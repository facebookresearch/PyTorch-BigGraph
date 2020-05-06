import torch
from torch.optim import Adagrad


class AsyncAdagrad(Adagrad):
    """Variant of Adagrad that is more robust to asynchronous (HOGWILD) updates.

    c.f. torch.optim.Adagrad
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(p, alpha=group["weight_decay"])

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if grad.is_sparse:
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # multiple HOGWILD processes may perform unsynchronized
                    # updates to G. Update a local copy of G independently from
                    # the shared-memory copy, to guarantee that
                    # local_G >= grad^2
                    local_G = state["sum"].sparse_mask(grad)._values()
                    delta_G = grad_values.pow(2)
                    state["sum"].add_(make_sparse(delta_G))
                    local_G += delta_G
                    std_values = local_G.sqrt_().add_(group["eps"])
                    p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    # multiple HOGWILD processes may perform unsynchronized
                    # updates to G. Update a local copy of G independently from
                    # the shared-memory copy, to guarantee that
                    # local_G >= grad^2
                    local_G = state["sum"].clone()
                    delta_G = grad * grad
                    state["sum"].add_(delta_G)
                    local_G += delta_G
                    std = local_G.sqrt().add_(group["eps"])
                    p.addcdiv_(grad, std, value=-clr)

        return loss
