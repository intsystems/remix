"""Objectives using the public sampling and density interface."""

from typing import Callable

from torch import Tensor

from .distributions import Mixture


class ELBOObjective:
    """ELBO integrand, log_joint and log_prob must have identical shapes."""

    def __init__(self, log_joint: Callable[[Tensor], Tensor], stl: bool = False):
        if stl:
            raise NotImplementedError("STL is not implemented in this score-function proof of concept.")
        self.log_joint = log_joint

    def evaluate(self, q: Mixture, z: Tensor) -> Tensor:
        log_q = q.log_prob(z)
        log_p = self.log_joint(z)
        if log_p.shape != log_q.shape:
            raise ValueError("log_joint and log_prob must have identical shapes, use Independent for vector events.")
        return log_p - log_q
