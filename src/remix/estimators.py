"""Gradient estimators returning a minimizable surrogate loss."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .distributions import Mixture
from .objectives import ELBOObjective


@dataclass
class EstimateResult:
    value: Tensor
    surrogate_loss: Tensor


class GradientEstimator(ABC):
    def __init__(self, n_samples: int = 1):
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        
        self.n_samples = n_samples

    @abstractmethod
    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate an objective and construct its negative gradient surrogate."""


class ScoreFunction(GradientEstimator):
    """REINFORCE algorithm with a fixed, sample-independent baseline."""

    def __init__(self, n_samples: int = 1, baseline: float = 0.0):
        super().__init__(n_samples)
        self.baseline = float(baseline)

    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        if not isinstance(q, Mixture):
            raise TypeError("q must be a Mixture.")

        z = q.sample(torch.Size([self.n_samples])).detach()
        log_q = q.log_prob(z)
        values = objective.evaluate(q, z)
        
        surrogate = (values.detach() - self.baseline) * log_q + values
        return EstimateResult(values.detach().mean(), -surrogate.mean())
