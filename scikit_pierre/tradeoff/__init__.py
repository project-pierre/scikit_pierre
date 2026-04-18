"""
Tradeoff sub-package.

Provides calibrated recommendation re-ranking algorithms that balance
relevance and genre-distribution calibration when producing a final
recommendation list from a set of candidate items.

Classes
-------
LinearCalibration
    Greedy surrogate optimisation (Silva et al., 2021; Steck, 2018).
LogarithmBias
    Linear calibration with an item-bias correction term.
PopularityCalibration
    LinearCalibration variant for popularity-group item sets.
TwoStageCalibration
    Two-stage pipeline: popularity calibration then genre calibration.
"""
