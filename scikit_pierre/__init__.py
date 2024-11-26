"""
Scikit-Pierre is a Scientific ToolKit for Post-processing Recommendations.
"""
from .tradeoff import calibration
from .metrics import evaluation
from .distributions import compute_distribution

__version__ = "0.0.2"

__all__ = ["calibration", "evaluation", "compute_distribution", "__version__"]
