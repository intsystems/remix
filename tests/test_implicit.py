"""Tests for implicit reparameterization of mixture samples."""

import pytest
import torch
from torch.distributions import (
    Bernoulli,
    Independent,
    Laplace,
    MultivariateNormal,
    Normal,
)

from remix import ELBOObjective, EstimateResult, Implicit, Mixture

DTYPE = torch.float64


@pytest.mark.parametrize("stl", [False, True], ids=["plain", "stl"])
@pytest.mark.parametrize("n_samples", [1, 7])
def test_implicit_scalar_gradient_matches_cdf_formula(stl, n_samples):
    logits = torch.tensor([-0.2, 0.3], dtype=DTYPE, requires_grad=True)
    locs = torch.tensor([-0.8, 0.6], dtype=DTYPE, requires_grad=True)
    scales = torch.tensor([0.9, 1.3], dtype=DTYPE, requires_grad=True)
    target_loc = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    params = (logits, locs, scales, target_loc)
    q = Mixture(
        logits, [Normal(locs[0], scales[0]), Laplace(locs[1], scales[1])]
    )
    objective = ELBOObjective(Normal(target_loc, 1.2).log_prob, stl=stl)

    torch.manual_seed(13)
    result = Implicit(n_samples=n_samples).estimate(q, objective)
    torch.manual_seed(13)
    z = q.sample((n_samples,)).requires_grad_()
    values = objective.evaluate(q, z)

    assert isinstance(result, EstimateResult)
    assert not result.value.requires_grad
    torch.testing.assert_close(result.value, values.detach().mean())
    torch.testing.assert_close(result.surrogate_loss.detach(), -result.value)

    # For a scalar sample, dz/dtheta = -dF/dtheta / q(z).
    (path_grad,) = torch.autograd.grad(values.sum(), z, retain_graph=True)
    multiplier = (path_grad / q.log_prob(z).exp()).detach()
    cdfs = torch.stack([c.cdf(z) for c in q.components], dim=-1)
    mixture_cdf = cdfs @ logits.softmax(dim=-1)
    expected_loss = -(values - multiplier * mixture_cdf).mean()
    expected = torch.autograd.grad(expected_loss, params, retain_graph=True)
    actual = torch.autograd.grad(result.surrogate_loss, params)

    for grad, reference in zip(actual, expected):
        torch.testing.assert_close(grad, reference)


def test_implicit_vector_samples_hold_conditional_cdfs_fixed():
    logits = torch.tensor([-0.4, 0.2], dtype=DTYPE, requires_grad=True)
    locs = torch.tensor(
        [[-0.8, 0.2, 0.7], [0.5, -0.6, -0.1]], dtype=DTYPE, requires_grad=True
    )
    scales = torch.tensor(
        [[0.8, 1.0, 1.1], [1.1, 0.9, 0.8]], dtype=DTYPE, requires_grad=True
    )
    components = [
        Independent(Normal(loc, scale), 1) for loc, scale in zip(locs, scales)
    ]
    q = Mixture(logits, components)
    observed = []

    def log_joint(z):
        observed.append(z)
        return -z.square().sum(dim=-1)

    torch.manual_seed(7)
    try:
        Implicit(n_samples=3).estimate(q, ELBOObjective(log_joint))
    except TypeError as exc:
        if str(exc) == "Implicit currently requires scalar-event components.":
            pytest.xfail(str(exc))
        raise
    (z,) = observed
    torch.manual_seed(7)
    torch.testing.assert_close(z, q.sample((3,)), rtol=0, atol=0)

    marginals = Normal(locs, scales)
    cdfs = marginals.cdf(z.unsqueeze(1))
    log_densities = marginals.log_prob(z.unsqueeze(1))
    for i in range(z.shape[-1]):
        weights = (logits + log_densities[..., :i].sum(dim=-1)).softmax(dim=-1)
        conditional_cdf = (weights * cdfs[..., i]).sum(dim=-1)
        for probability in conditional_cdf:
            grads = torch.autograd.grad(
                probability, (logits, locs, scales), retain_graph=True
            )
            for grad in grads:
                torch.testing.assert_close(
                    grad, torch.zeros_like(grad), rtol=0, atol=1e-10
                )


def test_implicit_with_stl_has_zero_gradient_at_the_target():
    logits = torch.tensor([-0.3, 0.1], dtype=DTYPE, requires_grad=True)
    locs = torch.tensor([-0.6, 0.9], dtype=DTYPE, requires_grad=True)
    scales = torch.tensor([0.8, 1.2], dtype=DTYPE, requires_grad=True)
    components = [Normal(loc, scale) for loc, scale in zip(locs, scales)]
    q = Mixture(logits, components)
    objective = ELBOObjective(q.detach().log_prob, stl=True)

    torch.manual_seed(11)
    result = Implicit(n_samples=8).estimate(q, objective)
    grads = torch.autograd.grad(result.surrogate_loss, (logits, locs, scales))

    torch.testing.assert_close(result.value, torch.zeros((), dtype=DTYPE))
    for grad in grads:
        torch.testing.assert_close(
            grad, torch.zeros_like(grad), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize(
    ("component", "error_pattern"),
    [
        (Bernoulli(0.5), r"continuous components with rsample\(\)"),
        (
            MultivariateNormal(torch.zeros(2), torch.eye(2)),
            "scalar-event components",
        ),
    ],
    ids=["discrete", "vector_event"],
)
def test_implicit_requires_scalar_continuous_cdfs(component, error_pattern):
    q = Mixture(torch.zeros(1), [component])
    with pytest.raises(TypeError, match=error_pattern):
        Implicit().estimate(q, ELBOObjective(q.log_prob))
