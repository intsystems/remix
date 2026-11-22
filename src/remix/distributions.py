"""Finite mixtures with arbitrary compatible component families."""
 
from collections.abc import Sequence
 
import torch
from torch import Tensor
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
)
 
# Alternative parameterisations of one and the same distribution. A
# constructor accepts only one of them, so ``detach_distribution`` keeps the
# first name of a group that it finds on the instance.
_EXCLUSIVE_PARAM_GROUPS = (
    ("logits", "probs"),
    ("scale_tril", "covariance_matrix", "precision_matrix"),
)
 
 
def detach_distribution(dist: Distribution) -> Distribution:
    """Rebuild ``dist`` with all parameters cut from the autograd graph.
 
    The result has the same density, but no gradient flows into its
    parameters. A sample ``z`` passed to its ``log_prob`` keeps its own
    gradient, which is what the Sticking-the-Landing trick needs.
 
    Supports ``Independent``, ``MixtureSameFamily``, any distribution that
    defines its own ``detach()`` and every distribution built from the
    tensors listed in ``arg_constraints`` (``Normal``, ``StudentT``,
    ``Laplace``, ``MultivariateNormal``, ``Categorical``, ...).
    """
    # A distribution may know how to detach itself (custom components).
    if hasattr(dist, "detach") and callable(dist.detach):
        return dist.detach()
 
    validate_args = dist._validate_args
 
    # Wrappers: detach what they wrap, then rebuild the wrapper itself.
    if isinstance(dist, Independent):
        return Independent(
            detach_distribution(dist.base_dist),
            dist.reinterpreted_batch_ndims,
            validate_args=validate_args,
        )
    if isinstance(dist, MixtureSameFamily):
        return MixtureSameFamily(
            detach_distribution(dist.mixture_distribution),
            detach_distribution(dist.component_distribution),
            validate_args=validate_args,
        )
 
    # Plain distributions: take the parameters stored on the instance. A
    # lazily computed alternative (``probs`` next to ``logits``) may be
    # cached there as well, so only one name per exclusive group is kept.
    names = [
        name
        for name in getattr(dist, "arg_constraints", {})
        if name in dist.__dict__
    ]
    for group in _EXCLUSIVE_PARAM_GROUPS:
        present = [name for name in group if name in names]
        for redundant in present[1:]:
            names.remove(redundant)
    if not names:
        raise NotImplementedError(
            f"Cannot detach {type(dist).__name__}: "
            "wrap it or give it a detach() method."
        )
 
    params = {name: getattr(dist, name).detach() for name in names}
    return type(dist)(**params, validate_args=validate_args)
 
 
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
        return torch.logsumexp(self._log_joint(z), dim=-1)
 
    def responsibilities(self, z: Tensor) -> Tensor:
        """Return posterior component probabilities for each sample."""
        return self._log_joint(z).softmax(dim=-1)
 
    def detach(self) -> "Mixture":
        """Return a copy with the same density and parameters without grad.
 
        Gradients still flow through the argument ``z`` of ``log_prob``, but
        not into the weights and component parameters (the "stop gradient"
        of STL).
        """
        return Mixture(
            self.logits.detach(),
            [detach_distribution(c) for c in self.components],
        )
 