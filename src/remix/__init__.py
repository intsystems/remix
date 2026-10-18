"""Monte Carlo gradient estimation for finite mixtures."""

from .distributions import Mixture
from .objectives import ELBOObjective
from .estimators import EstimateResult, GradientEstimator, ScoreFunction

__version__ = "1.0.0"
__all__ = ["Mixture", "ELBOObjective", "EstimateResult", "GradientEstimator", "ScoreFunction"]
