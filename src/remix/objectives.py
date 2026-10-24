"""Objectives using the public sampling and density interface."""

from typing import Callable

from torch import Tensor

from .distributions import Mixture


class ELBOObjective:
    """ELBO integrand, log_joint and log_prob must have identical shapes.

    The ELBO is `E_q[log p(z) - log q(z)]`; this class evaluates the integrand
    at given samples.

    Args:
        log_joint: Unnormalized target log density `log p(z)`, mapping points
            of shape `sample_shape + event_shape` to `sample_shape`.
        stl: Use the sticking-the-landing estimator. Not implemented yet.

    Raises:
        NotImplementedError: If `stl` is `True`.
    """

    def __init__(self, log_joint: Callable[[Tensor], Tensor], stl: bool = False):
        if stl:
            raise NotImplementedError("STL is not implemented in this score-function proof of concept.")
        self.log_joint = log_joint

    def evaluate(self, q: Mixture, z: Tensor) -> Tensor:
        """Evaluate `log p(z) - log q(z)` at each sample.

        Args:
            q: Variational mixture.
            z: Samples of shape `sample_shape + event_shape`.

        Returns:
            Integrand values of shape `sample_shape`.

        Raises:
            ValueError: If `log_joint(z)` and `q.log_prob(z)` differ in shape.
        """
        log_q = q.log_prob(z)
        log_p = self.log_joint(z)
        if log_p.shape != log_q.shape:
            raise ValueError("log_joint and log_prob must have identical shapes, use Independent for vector events.")
        return log_p - log_q
