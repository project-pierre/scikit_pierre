"""
Distributions sub-package.

Provides functions that compute a per-user genre/class probability
distribution from a set of :class:`~scikit_pierre.models.item.Item`
instances.  Distributions are used as the "target" (from user history) and
"realized" (from the current recommendation list) vectors in the calibration
trade-off objective.

Variants
--------
class_based
    Score-weighted genre proportions (CWS, WPS, PGD, PGD_P).
time_based
    Timestamp-discounted genre proportions (TWB, TWB_P, TGD, TGD_P).
entropy_based
    Entropy-weighted genre proportions (GLEB, GLEB_P).
mixed_based
    Combinations of the above (TWB_GLEB, TWB_GLEB_P).
time_slide_window_based
    Sliding-window temporal variants (TSW family).
"""
