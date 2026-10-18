"""Finite mixtures with arbitrary compatible component families."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution


class Mixture:
    """An unbatched mixture, components share event shape and base measure."""

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
        sample_shape = torch.tensor(sample_shape)
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
        return torch.logsumexp(self._log_joint(z), dim=-1)

    def responsibilities(self, z: Tensor) -> Tensor:
        """Return posterior component probabilities for each sample."""
        return self._log_joint(z).softmax(dim=-1)
