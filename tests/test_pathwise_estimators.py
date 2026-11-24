import pytest
import torch
from torch.distributions import Independent, Normal

from remix import (
    ELBOObjective,
    Implicit,
    Mixture,
    PostStratified,
    ScoreFunction,
    Stratified,
)


DTYPE = torch.float64


@pytest.mark.parametrize(
    "estimator_type",
    [ScoreFunction, Stratified, Implicit, PostStratified],
)
def test_all_estimators_share_the_public_interface(estimator_type):
    logits = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    q = Mixture(
        logits,
        [
            Normal(torch.tensor(-1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)),
            Normal(torch.tensor(1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)),
        ],
    )
    objective = ELBOObjective(Normal(0.0, 1.0).log_prob)

    result = estimator_type(n_samples=4).estimate(q, objective)

    assert result.value.shape == result.surrogate_loss.shape == ()
    assert not result.value.requires_grad
    result.surrogate_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("estimator_type", [Implicit, PostStratified])
def test_scalar_normal_gradient_matches_exact_elbo(estimator_type):
    torch.manual_seed(4)
    loc = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    scale = torch.tensor(1.3, dtype=DTYPE, requires_grad=True)
    logits = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    q = Mixture(logits, [Normal(loc, scale)])
    target = Normal(torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    result = estimator_type(n_samples=8192).estimate(
        q, ELBOObjective(target.log_prob)
    )

    result.surrogate_loss.backward()

    assert torch.allclose(loc.grad, loc.detach(), atol=0.04, rtol=0.0)
    expected_scale_grad = scale.detach() - scale.detach().reciprocal()
    assert torch.allclose(scale.grad, expected_scale_grad, atol=0.04, rtol=0.0)
    assert result.value.shape == result.surrogate_loss.shape == ()
    assert torch.isfinite(result.value)
    assert torch.isfinite(result.surrogate_loss)


@pytest.mark.parametrize("estimator_type", [Implicit, PostStratified])
def test_estimators_reject_vector_events(estimator_type):
    q = Mixture(
        torch.zeros(1, dtype=DTYPE),
        [
            Independent(
                Normal(torch.zeros(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE)),
                1,
            )
        ],
    )
    objective = ELBOObjective(lambda z: -z.square().sum(dim=-1))

    with pytest.raises(TypeError, match="scalar-event"):
        estimator_type().estimate(q, objective)


@pytest.mark.parametrize("estimator_type", [Implicit, PostStratified])
def test_mixture_logit_gradient_matches_quadrature(estimator_type):
    logits = torch.tensor([0.2, -0.1], dtype=DTYPE, requires_grad=True)
    components = [
        Normal(torch.tensor(-1.0, dtype=DTYPE), torch.tensor(0.8, dtype=DTYPE)),
        Normal(torch.tensor(1.2, dtype=DTYPE), torch.tensor(1.1, dtype=DTYPE)),
    ]
    target = Normal(torch.tensor(0.3, dtype=DTYPE), torch.tensor(1.2, dtype=DTYPE))
    objective = ELBOObjective(target.log_prob)

    q = Mixture(logits, components)
    grid = torch.linspace(-10.0, 10.0, 20001, dtype=DTYPE)
    log_q = q.log_prob(grid)
    exact_elbo = torch.trapz(
        log_q.exp() * (target.log_prob(grid) - log_q), grid
    )
    (exact_gradient,) = torch.autograd.grad(exact_elbo, logits)

    torch.manual_seed(7)
    q = Mixture(logits, components)
    result = estimator_type(n_samples=4096).estimate(q, objective)
    (estimated_loss_gradient,) = torch.autograd.grad(
        result.surrogate_loss, logits
    )

    assert torch.allclose(
        estimated_loss_gradient, -exact_gradient, atol=0.06, rtol=0.0
    )
