# ReMix

ReMix is a PyTorch library for Monte Carlo gradient estimation with finite
mixture distributions.

The public API provides a shared `GradientEstimator.estimate(q, objective)`
interface for:

- `ScoreFunction`: score-function estimator with a constant baseline.
- `Stratified`: reparameterized samples from every mixture component.
- `Implicit`: implicit CDF-rank estimator for scalar continuous mixtures.
- `PostStratified`: continuous-transport estimator using component CDFs and
  inverse CDFs.

`Implicit` requires differentiable component `cdf()` methods.
`PostStratified` also requires differentiable `icdf()` methods. Both currently
support scalar-event components; `Stratified` requires every component to
support `rsample()`.

See the [documentation](docs/source/index.md) for installation, examples, and
API details.
