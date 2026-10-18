import torch
from torch.distributions import Laplace, Normal

from remix import ELBOObjective, Mixture
from remix.estimators import ScoreFunction


def fit(name, components):
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.zeros(len(components)))
    target = Mixture(torch.tensor([0.2, 0.8]).log(), components)
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    objective = ELBOObjective(target.log_prob)
    estimator = ScoreFunction(n_samples=512)
    def measure_kl():
        with torch.no_grad():
            q = Mixture(parameter, components)
            z = q.sample(torch.Size([10000]))
            return (q.log_prob(z) - target.log_prob(z)).mean().item()

    initial = measure_kl()
    for _ in range(300):
        optimizer.zero_grad()
        result = estimator.estimate(Mixture(parameter, components), objective)
        result.surrogate_loss.backward()
        optimizer.step()
    final = measure_kl()
    print(f"{name}: estimated KL {initial:.4f} -> {final:.4f}")
    print(f"Learned weights: {parameter.detach().softmax(-1).tolist()}")
    return initial, final


if __name__ == "__main__":
    fit("Normal mixture", [Normal(-2.0, 1.0), Normal(2.0, 1.0)])
    fit("Laplace mixture", [Laplace(-2.0, 1.0), Laplace(2.0, 1.0)])
    fit("Normal + Laplace mixture", [Normal(-2.0, 1.0), Laplace(2.0, 1.0)])
