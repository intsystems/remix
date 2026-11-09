"""Tests of the Sticking-the-Landing (STL) trick.

Covers ``detach_distribution`` / ``Mixture.detach``, ``ELBOObjective(stl=True)``
and ``ScoreFunction`` used together with it.
"""

import pytest
import torch
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    StudentT,
)

from remix import ELBOObjective, EstimateResult, Mixture, ScoreFunction
from remix.distributions import detach_distribution

DTYPE = torch.float64


def make_q(logits, locs, log_scales):
    """1-D Gaussian mixture; ``locs`` and ``log_scales`` have shape (K, 1)."""
    components = [
        Independent(Normal(locs[k], log_scales[k].exp()), 1) for k in range(len(logits))
    ]
    return Mixture(logits, components)


def make_params(seed=0, k=3):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(k, generator=g, dtype=DTYPE)
    locs = torch.randn(k, 1, generator=g, dtype=DTYPE)
    log_scales = 0.2 * torch.randn(k, 1, generator=g, dtype=DTYPE)
    return [t.requires_grad_() for t in (logits, locs, log_scales)]


@pytest.fixture
def target():
    """Fixed target density p(z): a two-component 1-D mixture."""
    logits = torch.tensor([0.3, -0.2], dtype=DTYPE)
    locs = torch.tensor([[-1.5], [1.2]], dtype=DTYPE)
    log_scales = torch.log(torch.tensor([[0.8], [1.1]], dtype=DTYPE))
    return make_q(logits, locs, log_scales)


def exact_neg_elbo_grad(params, target):
    """Gradient of -ELBO computed by quadrature on a fine 1-D grid."""
    q = make_q(*params)
    grid = torch.linspace(-15.0, 15.0, 30001, dtype=DTYPE).unsqueeze(-1)
    log_q, log_p = q.log_prob(grid), target.log_prob(grid)
    neg_elbo = -torch.trapz(log_q.exp() * (log_p - log_q), grid.squeeze(-1))
    grads = torch.autograd.grad(neg_elbo, params)
    return torch.cat([g.reshape(-1) for g in grads])


def gradient_moments(loss_fn, params, n_repeats):
    """Monte Carlo mean and variance of the gradient of ``loss_fn()``."""
    draws = []
    for _ in range(n_repeats):
        grads = torch.autograd.grad(loss_fn(), params, allow_unused=True)
        draws.append(
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, grads)
                ]
            )
        )
    stacked = torch.stack(draws)
    return stacked.mean(0), stacked.var(0, unbiased=True)


# --------------------------------------------------------------------------
# detach_distribution
# --------------------------------------------------------------------------
def _grad_blocked(dist, params, z):
    """log_prob is unchanged, depends on no parameter, and keeps its z-path."""
    detached = detach_distribution(dist)
    z = z.clone().requires_grad_()
    out = detached.log_prob(z)
    assert torch.allclose(out, dist.log_prob(z))
    grads = torch.autograd.grad(out.sum(), [z] + list(params), allow_unused=True)
    assert grads[0].abs().sum() > 0
    assert all(g is None for g in grads[1:])


def test_detach_normal_independent():
    loc = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    scale = torch.ones(2, dtype=DTYPE, requires_grad=True)
    _grad_blocked(Independent(Normal(loc, scale), 1), [loc, scale], torch.randn(5, 2, dtype=DTYPE))


def test_detach_student_t_heterogeneous_family():
    df = torch.tensor([4.0], dtype=DTYPE, requires_grad=True)
    loc = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    scale = torch.ones(1, dtype=DTYPE, requires_grad=True)
    _grad_blocked(Independent(StudentT(df, loc, scale), 1), [df, loc, scale], torch.randn(5, 1, dtype=DTYPE))


def test_detach_multivariate_normal_covariance_parameterisation():
    loc = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    cov = (2.0 * torch.eye(2, dtype=DTYPE)).requires_grad_()
    _grad_blocked(MultivariateNormal(loc, covariance_matrix=cov), [loc, cov], torch.randn(5, 2, dtype=DTYPE))


def test_detach_mixture_same_family():
    logits = torch.randn(3, dtype=DTYPE, requires_grad=True)
    loc = torch.randn(3, 1, dtype=DTYPE, requires_grad=True)
    dist = MixtureSameFamily(
        Categorical(logits=logits),
        Independent(Normal(loc, torch.ones(3, 1, dtype=DTYPE)), 1),
    )
    _grad_blocked(dist, [logits, loc], torch.randn(5, 1, dtype=DTYPE))


def test_detach_categorical_with_both_parameterisations_cached():
    logits = torch.randn(4, dtype=DTYPE, requires_grad=True)
    cat = Categorical(logits=logits)
    _ = cat.probs  # caches probs next to logits
    detached = detach_distribution(cat)
    assert torch.allclose(detached.probs, cat.probs)
    assert not detached.probs.requires_grad


def test_detach_uses_custom_detach_method():
    class Custom(Distribution):
        arg_constraints = {}

        def detach(self):
            return "custom"

    assert detach_distribution(Custom(validate_args=False)) == "custom"


def test_detach_unsupported_distribution_raises():
    class Opaque(Distribution):
        arg_constraints = {}

    with pytest.raises(NotImplementedError):
        detach_distribution(Opaque(validate_args=False))


# --------------------------------------------------------------------------
# Mixture.detach
# --------------------------------------------------------------------------
def test_mixture_detach_keeps_density_and_blocks_parameter_gradients():
    params = make_params()
    q = make_q(*params)
    qd = q.detach()
    z = torch.randn(7, 1, dtype=DTYPE)

    assert isinstance(qd, Mixture) and qd is not q
    assert torch.allclose(qd.log_prob(z), q.log_prob(z))
    assert torch.allclose(qd.responsibilities(z), q.responsibilities(z))
    assert not qd.log_prob(z).requires_grad
    assert q.log_prob(z).requires_grad  # the original is untouched


def test_mixture_detach_supports_mixed_component_families():
    logits = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    loc = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    q = Mixture(
        logits,
        [
            Independent(Normal(loc, torch.ones(1, dtype=DTYPE)), 1),
            Independent(StudentT(torch.tensor([3.0], dtype=DTYPE), loc, torch.ones(1, dtype=DTYPE)), 1),
        ],
    )
    z = torch.randn(4, 1, dtype=DTYPE)
    qd = q.detach()
    assert torch.allclose(qd.log_prob(z), q.log_prob(z))
    assert not qd.log_prob(z).requires_grad


# --------------------------------------------------------------------------
# ELBOObjective(stl=True)
# --------------------------------------------------------------------------
def test_stl_is_accepted_and_does_not_change_values(target):
    params = make_params()
    z = torch.randn(5, 1, dtype=DTYPE)
    plain = ELBOObjective(target.log_prob, stl=False).evaluate(make_q(*params), z)
    stl = ELBOObjective(target.log_prob, stl=True).evaluate(make_q(*params), z)
    assert torch.allclose(plain, stl)


def test_stl_drops_density_gradient_but_keeps_path_gradient(target):
    params = make_params()
    z = torch.randn(5, 1, dtype=DTYPE, requires_grad=True)

    stl = ELBOObjective(target.log_prob, stl=True).evaluate(make_q(*params), z)
    grads = torch.autograd.grad(stl.sum(), [z] + params, allow_unused=True)
    assert grads[0].abs().sum() > 0
    assert all(g is None for g in grads[1:])

    plain = ELBOObjective(target.log_prob, stl=False).evaluate(make_q(*params), z)
    grads = torch.autograd.grad(plain.sum(), params, allow_unused=True)
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_stl_objective_keeps_gradients_of_model_parameters():
    phi = torch.tensor([0.5], dtype=DTYPE, requires_grad=True)
    q = make_q(*make_params())
    z = q.sample((8,))
    value = ELBOObjective(lambda x: Normal(phi, 1.0).log_prob(x).sum(-1), stl=True).evaluate(q, z)
    (g,) = torch.autograd.grad(value.sum(), [phi])
    assert g.abs().sum() > 0


def test_objective_shape_mismatch_still_raises():
    q = make_q(*make_params())
    objective = ELBOObjective(lambda z: z, stl=True)  # shape (N, 1) vs (N,)
    with pytest.raises(ValueError):
        objective.evaluate(q, q.sample((4,)))


# --------------------------------------------------------------------------
# ScoreFunction with STL
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stl", [False, True])
def test_estimate_returns_scalar_result_and_finite_gradients(target, stl):
    params = make_params()
    result = ScoreFunction(n_samples=8).estimate(
        make_q(*params), ELBOObjective(target.log_prob, stl=stl)
    )
    assert isinstance(result, EstimateResult)
    assert result.surrogate_loss.shape == () and result.value.shape == ()
    assert not result.value.requires_grad
    result.surrogate_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_stl_surrogate_is_plain_reinforce_form(target):
    """With STL the gradient is exactly mean((L - b) * grad log q)."""
    params, n, baseline = make_params(), 16, 0.3

    torch.manual_seed(11)
    result = ScoreFunction(n_samples=n, baseline=baseline).estimate(
        make_q(*params), ELBOObjective(target.log_prob, stl=True)
    )
    got = torch.autograd.grad(result.surrogate_loss, params)

    torch.manual_seed(11)
    q = make_q(*params)
    z = q.sample(torch.Size([n]))
    log_q = q.log_prob(z)
    advantage = (target.log_prob(z) - log_q).detach() - baseline
    expected = torch.autograd.grad(-(advantage * log_q).mean(), params)

    for g, e in zip(got, expected):
        assert torch.allclose(g, e)


def test_plain_surrogate_keeps_the_grad_l_term(target):
    """Without STL the extra term -grad log q of the ELBO is present."""
    params, n = make_params(), 16

    torch.manual_seed(5)
    result = ScoreFunction(n_samples=n).estimate(
        make_q(*params), ELBOObjective(target.log_prob, stl=False)
    )
    got = torch.autograd.grad(result.surrogate_loss, params)

    torch.manual_seed(5)
    q = make_q(*params)
    z = q.sample(torch.Size([n]))
    log_q = q.log_prob(z)
    values = target.log_prob(z) - log_q
    expected = torch.autograd.grad(-(values.detach() * log_q + values).mean(), params)

    for g, e in zip(got, expected):
        assert torch.allclose(g, e)


def test_plain_equals_stl_with_baseline_shifted_by_one(target):
    """For an ELBO, the grad L term equals -grad log q, so plain == STL with b + 1."""
    params, n, baseline = make_params(), 16, 0.3

    def gradients(stl, b):
        torch.manual_seed(21)
        result = ScoreFunction(n_samples=n, baseline=b).estimate(
            make_q(*params), ELBOObjective(target.log_prob, stl=stl)
        )
        return torch.autograd.grad(result.surrogate_loss, params)

    for g, e in zip(gradients(False, baseline), gradients(True, baseline + 1.0)):
        assert torch.allclose(g, e)


@pytest.mark.parametrize("stl", [False, True], ids=["plain", "stl"])
@pytest.mark.parametrize("baseline", [0.0, -1.0], ids=["b0", "b-1"])
def test_score_function_is_unbiased(target, stl, baseline):
    """Mean of the estimator matches the exact gradient of -ELBO (quadrature)."""
    torch.manual_seed(123)
    params = make_params(seed=5)
    exact = exact_neg_elbo_grad(params, target)
    estimator = ScoreFunction(n_samples=64, baseline=baseline)
    objective = ELBOObjective(target.log_prob, stl=stl)

    n_repeats = 400
    mean, var = gradient_moments(
        lambda: estimator.estimate(make_q(*params), objective).surrogate_loss,
        params,
        n_repeats,
    )
    std_err = (var / n_repeats).sqrt()
    assert torch.all((mean - exact).abs() <= 5 * std_err + 1e-6), (mean, exact, std_err)
