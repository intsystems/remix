"""Objectives using the public sampling and density interface."""
 
from typing import Callable
 
from torch import Tensor
 
from .distributions import Mixture
 
 
class ELBOObjective:
    """ELBO integrand, log_joint and log_prob must have identical shapes.
 
    With ``stl=True`` the Sticking-the-Landing trick of Roeder et al. (2017)
    is applied: ``log q(z)`` is evaluated with the parameters of ``q``
    detached ("stop gradient"). The values of the integrand are unchanged,
    but its explicit dependence on the parameters of ``q`` (a score term
    with zero mean) disappears from the gradient. The gradient that flows
    through the sample ``z`` itself is kept, so a reparameterised ``z``
    still produces the pathwise gradient, and gradients w.r.t. parameters
    of ``log_joint`` are unaffected. Estimators remain unbiased; the
    variance goes to zero as ``q`` approaches the true posterior.

    With ``stl=True`` the Sticking-the-Landing trick of Roeder et al. (2017)
    is applied: ``log q(z)`` is evaluated with the parameters of ``q``
    detached ("stop gradient"). The values of the integrand are unchanged, but
    its explicit dependence on the parameters of ``q`` (a score term with zero
    mean) disappears from the gradient. The gradient that flows through the
    sample ``z`` itself is kept, so a reparameterised ``z`` still produces the
    pathwise gradient, and gradients w.r.t. parameters of ``log_joint`` are
    unaffected. Estimators remain unbiased; the variance goes to zero as ``q``
    approaches the true posterior.
    """
 
    def __init__(
        self,
        log_joint: Callable[[Tensor], Tensor],
        stl: bool = False,
    ):
        self.log_joint = log_joint
        self.stl = bool(stl)
 
    def evaluate(self, q: Mixture, z: Tensor) -> Tensor:
        # With STL the density is evaluated by a copy of q without gradient
        # into its parameters; only the path through z is differentiable.
        log_q = (q.detach() if self.stl else q).log_prob(z)
        log_p = self.log_joint(z)
        if log_p.shape != log_q.shape:
            raise ValueError(
                "log_joint and log_prob must have identical shapes, "
                "use Independent for vector events."
            )
        return log_p - log_q
 