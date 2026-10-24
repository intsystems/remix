# ReMix

Monte Carlo gradient estimation for finite mixture distributions in PyTorch.

ReMix fits a variational mixture `q(z) = sum_k w_k p_k(z)` to a target density by
maximizing the ELBO. Components may come from different families
(`Normal`, `Laplace`, ...) and need not be reparameterizable.

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0

### From source

```
git clone https://github.com/intsystems/remix.git
python -m pip install ./remix/src
```

**Uninstall**

```
python -m pip uninstall remix
```

## Quick start

```python
import torch
from torch.distributions import Laplace, Normal

from remix import ELBOObjective, Mixture, ScoreFunction

components = [Normal(-2.0, 1.0), Laplace(2.0, 1.0)]
target = Mixture(torch.tensor([0.2, 0.8]).log(), components)

logits = torch.nn.Parameter(torch.zeros(len(components)))
optimizer = torch.optim.Adam([logits], lr=0.05)
objective = ELBOObjective(target.log_prob)
estimator = ScoreFunction(n_samples=512)

for _ in range(300):
    optimizer.zero_grad()
    result = estimator.estimate(Mixture(logits, components), objective)
    result.surrogate_loss.backward()
    optimizer.step()

print(logits.detach().softmax(-1))  # close to [0.2, 0.8]
```

A runnable version is in `examples/benchmark.py`.

## Main classes

| Class | Purpose |
|---|---|
| [`Mixture`](api/distributions.md) | Mixture density: sampling, `log_prob`, responsibilities. |
| [`ELBOObjective`](api/objectives.md) | ELBO integrand `log p(z) - log q(z)`. |
| [`ScoreFunction`](api/estimators.md) | REINFORCE gradient estimator with a constant baseline. |
