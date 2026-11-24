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


def _validate_scalar_continuous_mixture(q: Mixture, estimator: str) -> None:
    if q.event_shape:
        raise TypeError(f"{estimator} currently requires scalar-event components.")
    if any(not component.has_rsample for component in q.components):
        raise TypeError(f"{estimator} requires continuous components with rsample().")


def _sample_with_input_gradient(
    q: Mixture, objective: ELBOObjective, n_samples: int
) -> tuple[Tensor, Tensor, Tensor]:
    z = q.sample(torch.Size([n_samples])).detach().requires_grad_()
    values = objective.evaluate(q, z)
    if values.requires_grad:
        (input_gradient,) = torch.autograd.grad(
            values.sum(), z, retain_graph=True, allow_unused=True
        )
    else:
        input_gradient = None
    if input_gradient is None:
        input_gradient = torch.zeros_like(z)
    return z, values, input_gradient.detach()


class Implicit(GradientEstimator):
    """Implicit reparameterization estimator for scalar continuous mixtures.

    The mixture CDF defines a scalar rank map. Its parameter derivative gives
    the pathwise sampling term without differentiating through the sampled
    component label. Each component must provide a differentiable ``cdf``.

    Args:
        n_samples: Number of mixture samples per `estimate` call.
    """

    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate the objective using the mixture CDF rank map.

        Raises:
            TypeError: If the mixture is not scalar/continuous or its
                components do not implement ``cdf``.
            FloatingPointError: If inverse-density scaling is not finite.
        """
        if not isinstance(q, Mixture):
            raise TypeError("q must be a Mixture.")
        _validate_scalar_continuous_mixture(q, type(self).__name__)

        z, values, input_gradient = _sample_with_input_gradient(
            q, objective, self.n_samples
        )
        try:
            component_cdfs = torch.stack(
                [component.cdf(z.detach()) for component in q.components],
                dim=-1,
            )
        except NotImplementedError as exc:
            raise TypeError(
                "Implicit requires every component to implement cdf()."
            ) from exc

        mixture_cdf = (q.mixing.probs * component_cdfs).sum(dim=-1)
        inverse_density = torch.exp(-q.log_prob(z.detach()).detach())
        if not torch.isfinite(inverse_density).all():
            raise FloatingPointError(
                "Implicit encountered a non-finite inverse mixture density."
            )

        correction = -(
            input_gradient * inverse_density * mixture_cdf
        ).mean()
        surrogate = values.mean() + correction - correction.detach()
        return EstimateResult(values.detach().mean(), -surrogate)


class PostStratified(GradientEstimator):
    """Single-sample continuous-transport estimator for scalar mixtures.

    The estimator averages the component-label gradient conditional on each
    sampled point, using component responsibilities and inverse CDF maps.
    Components must provide differentiable ``cdf`` and ``icdf`` methods.

    Args:
        n_samples: Number of mixture samples per `estimate` call. The default
            of one is the single-sample estimator.
    """

    def estimate(self, q: Mixture, objective: ELBOObjective) -> EstimateResult:
        """Estimate the objective using post-stratified component transports.

        Raises:
            TypeError: If the mixture is not scalar/continuous or its
                components do not implement ``cdf`` and ``icdf``.
            FloatingPointError: If a component inverse CDF returns a
                non-finite transport point.
        """
        if not isinstance(q, Mixture):
            raise TypeError("q must be a Mixture.")
        _validate_scalar_continuous_mixture(q, type(self).__name__)

        z, values, input_gradient = _sample_with_input_gradient(
            q, objective, self.n_samples
        )
        transported_points = []
        try:
            for component in q.components:
                cdf = component.cdf(z.detach())
                transported_points.append(component.icdf(cdf.detach()))
        except NotImplementedError as exc:
            raise TypeError(
                "PostStratified requires every component to implement "
                "cdf() and icdf()."
            ) from exc

        transported = torch.stack(transported_points, dim=-1)
        if not torch.isfinite(transported).all():
            raise FloatingPointError(
                "PostStratified encountered a non-finite inverse-CDF point."
            )

        responsibilities = q.responsibilities(z.detach()).detach()
        component_samples = (
            input_gradient.unsqueeze(-1) * responsibilities * transported
        ).sum(dim=-1).mean()
        logits_gradient = (
            values.detach().unsqueeze(-1)
            * (responsibilities - q.mixing.probs.detach())
        ).mean(dim=0)
        logits_score = (q.logits * logits_gradient).sum()

        correction = component_samples + logits_score
        surrogate = values.mean() + correction - correction.detach()
        return EstimateResult(values.detach().mean(), -surrogate)
