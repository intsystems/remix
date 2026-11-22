"""Gradient estimators returning a minimizable surrogate loss."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .distributions import Mixture
from .objectives import ELBOObjective


@dataclass
class EstimateResult:
    """Output of a gradient estimator.

    Attributes:
        value: Detached Monte Carlo estimate of the objective, for logging.
        surrogate_loss: Scalar whose gradient is the negative gradient
            estimate; call `backward()` on it and step a minimizing optimizer.
    """

    value: Tensor
    surrogate_loss: Tensor


class GradientEstimator(ABC):
    """Base class for Monte Carlo gradient estimators.

    Args:
        n_samples: Number of samples drawn per `estimate` call.

    Raises:
        ValueError: If `n_samples` is not a positive integer.
    """

    def __init__(self, n_samples: int = 1):
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        
        self.n_samples = n_samples

    @abstractmethod
    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate an objective and construct its negative gradient surrogate.

        Args:
            q: Variational mixture whose parameters receive gradients.
            objective: Objective to maximize.

        Returns:
            The objective estimate and the surrogate loss.
        """


class ScoreFunction(GradientEstimator):
    """REINFORCE algorithm with a fixed, sample-independent baseline.

    Uses `grad E_q[f] = E_q[(f(z) - b) grad log q(z) + grad f(z)]` with samples
    detached from the parameters, so it works with non-reparameterizable
    components.

    Args:
        n_samples: Number of samples drawn per `estimate` call.
        baseline: Constant `b` subtracted from the objective values to reduce
            variance.

    Example:
        >>> import torch
        >>> from torch.distributions import Normal
        >>> from remix import ELBOObjective, Mixture
        >>> components = [Normal(-2.0, 1.0), Normal(2.0, 1.0)]
        >>> target = Mixture(torch.tensor([0.2, 0.8]).log(), components)
        >>> logits = torch.zeros(2, requires_grad=True)
        >>> estimator = ScoreFunction(n_samples=512)
        >>> objective = ELBOObjective(target.log_prob)
        >>> result = estimator.estimate(Mixture(logits, components), objective)
        >>> result.surrogate_loss.backward()
    """

    def __init__(self, n_samples: int = 1, baseline: float = 0.0):
        super().__init__(n_samples)
        self.baseline = float(baseline)

    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate the objective with the score-function surrogate.

        Args:
            q: Variational mixture whose parameters receive gradients.
            objective: Objective to maximize.

        Returns:
            The objective estimate and the surrogate loss.

        Raises:
            TypeError: If `q` is not a `Mixture`.
        """
        if not isinstance(q, Mixture):
            raise TypeError("q must be a Mixture.")

        z = q.sample(torch.Size([self.n_samples])).detach()
        log_q = q.log_prob(z)
        values = objective.evaluate(q, z)
        
        surrogate = (values.detach() - self.baseline) * log_q + values
        return EstimateResult(values.detach().mean(), -surrogate.mean())


class Stratified(GradientEstimator):
    """Pathwise gradient estimator with a fixed sample count per component.

    Draws independent reparameterized samples from each component and weights
    the component means by the mixture probabilities. With one sample per
    component this is Estimator 1 of Burroni and Sheldon (2026). Setting
    `stl=True` on the objective gives their Sticking-the-Landing variant.

    Args:
        n_samples: Number of samples per component per `estimate` call.
    """

    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate the objective by stratifying over the mixture components.

        Args:
            q: Variational mixture whose parameters receive gradients.
            objective: Objective to maximize.

        Returns:
            The objective estimate and its negative as the surrogate loss.

        Raises:
            TypeError: If `q` is not a `Mixture` or a component cannot be
                reparameterized.
        """
        if not isinstance(q, Mixture):
            raise TypeError("q must be a Mixture.")
        if any(not component.has_rsample for component in q.components):
            raise TypeError("Stratified requires all components to support rsample.")

        sample_shape = torch.Size([self.n_samples])
        z = torch.cat([component.rsample(sample_shape) for component in q.components])
        values = objective.evaluate(q, z).reshape(len(q.components), self.n_samples)
        estimate = (q.mixing.probs * values.mean(dim=1)).sum()
        return EstimateResult(estimate.detach(), -estimate)
