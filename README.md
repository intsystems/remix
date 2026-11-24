# ReMix

ReMix is a PyTorch library for Monte Carlo gradient estimation with finite
mixture distributions.

## Install

After a release is published to a package index:

```bash
python -m pip install remix-vi
```

To build and install a wheel from a source checkout:

```bash
python -m pip install build
python -m build
python -m pip install dist/remix_vi-1.0.0-py3-none-any.whl
```

The wheel can be copied to another project or uploaded to a package index; the
consumer does not need this repository. The PyPI distribution is named
`remix-vi`, while its Python import package remains `remix`. ReMix requires
Python 3.10+ and PyTorch 2.0+.

## Quick start

```python
import torch
from torch.distributions import Normal

from remix import ELBOObjective, Mixture, ScoreFunction

components = [Normal(-2.0, 1.0), Normal(2.0, 1.0)]
target = Mixture(torch.tensor([0.2, 0.8]).log(), components)
logits = torch.nn.Parameter(torch.zeros(2))
optimizer = torch.optim.Adam([logits], lr=0.05)
estimator = ScoreFunction(n_samples=512)
objective = ELBOObjective(target.log_prob)

optimizer.zero_grad()
result = estimator.estimate(Mixture(logits, components), objective)
result.surrogate_loss.backward()
optimizer.step()
```

The public API also includes `Stratified`, `Implicit`, and `PostStratified`.
The latter two currently require scalar-event components with differentiable
CDFs; `PostStratified` also requires differentiable inverse CDFs. `Stratified`
requires each component to support `rsample()`.

See the [API documentation](https://github.com/intsystems/remix/blob/master/docs/source/index.md).
