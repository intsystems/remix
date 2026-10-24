"""Finite mixtures with arbitrary compatible component families."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution


class Mixture:
    """An unbatched mixture, components share event shape and base measure.

    The density is `q(z) = sum_k softmax(logits)_k * p_k(z)`. Components may
    belong to different families (e.g. `Normal` and `Laplace`) as long as they
    are defined on the same space.

    Args:
        logits: Unnormalized log mixing weights of shape `(K,)`.
        components: `K` unbatched distributions with identical `event_shape`.
            A batched component such as `Normal(torch.zeros(d), 1.0)` must be
            wrapped in `torch.distributions.Independent(..., 1)` to become a
            single `d`-dimensional distribution.

    Raises:
        ValueError: If `components` is empty, `logits` is not one-dimensional,
            the number of logits differs from the number of components, or
            components are batched or have different event shapes.

    Example:
        >>> import torch
        >>> from torch.distributions import Normal, Laplace
        >>> q = Mixture(torch.zeros(2), [Normal(-2.0, 1.0), Laplace(2.0, 1.0)])
        >>> q.log_prob(q.sample((5,))).shape
        torch.Size([5])
    """

    def __init__(self, logits: Tensor, components: Sequence[Distribution]):
        self.components = tuple(components)

        if logits.ndim != 1 or logits.numel() != len(self.components) or not self.components:
            raise ValueError("Provide one logit per component in a nonempty mixture.")
        
        self.event_shape = self.components[0].event_shape

        if any(c.batch_shape or c.event_shape != self.event_shape for c in self.components):
            raise ValueError("Components must be unbatched and have identical event shapes.")
        
        self.logits = logits
        self.mixing = Categorical(logits=logits)

    def sample(self, sample_shape: tuple) -> Tensor:
        """Draw samples without a gradient path to the parameters.

        Args:
            sample_shape: Leading shape of the returned samples.

        Returns:
            Samples of shape `sample_shape + event_shape`.
        """
        sample_shape = torch.Size(sample_shape)
        count = sample_shape.numel()

        with torch.no_grad():
            indices = self.mixing.sample((count,))
            samples = torch.stack([c.sample((count,)) for c in self.components], dim=1)
            selected = samples[torch.arange(count, device=indices.device), indices]
            
            return selected.reshape(sample_shape + self.event_shape)

    def _log_joint(self, z: Tensor) -> Tensor:
        log_probs = torch.stack([c.log_prob(z) for c in self.components], dim=-1)
        return self.mixing.logits + log_probs

    def log_prob(self, z: Tensor) -> Tensor:
        """Compute the log density of the mixture.

        Args:
            z: Points of shape `sample_shape + event_shape`.

        Returns:
            Log densities of shape `sample_shape`, differentiable w.r.t. `logits`
            and component parameters.
        """
        return torch.logsumexp(self._log_joint(z), dim=-1)

    def responsibilities(self, z: Tensor) -> Tensor:
        """Return posterior component probabilities for each sample.

        Args:
            z: Points of shape `sample_shape + event_shape`.

        Returns:
            Probabilities `q(k | z)` of shape `sample_shape + (K,)`, summing to
            one over the last dimension.
        """
        return self._log_joint(z).softmax(dim=-1)
