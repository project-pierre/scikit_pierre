"""
Scikit-Pierre: Scientific ToolKit for Post-processing Recommendations.

This package provides calibrated recommendation re-ranking algorithms,
evaluation metrics, probability distribution functions, and statistical
similarity/divergence measures, all designed around a genre- or
class-based item representation.

Modules
-------
calibration
    Trade-off optimisation algorithms that balance relevance and fairness
    (calibration) when re-ordering a candidate recommendation list.
evaluation
    Accuracy, diversity, calibration, and unexpectedness metrics for
    offline evaluation of recommendation systems.
compute_distribution
    Utilities to compute per-user item-class probability distributions
    from raw interaction data.
"""
from .tradeoff import calibration
from .metrics import evaluation
from .distributions import compute_distribution

__version__ = "0.0.2"

__all__ = ["calibration", "evaluation", "compute_distribution", "__version__"]
