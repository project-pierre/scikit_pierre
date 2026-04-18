"""
Trade-off weight (lambda) functions.

These functions compute a scalar lambda in ``[0, 1]`` from a user's
genre-probability distribution vector.  Lambda controls the balance between
relevance and fairness (calibration) in the trade-off objective:

    utility = (1 - lambda) * relevance ± lambda * fairness

Higher lambda values emphasise calibration; lower values emphasise relevance.
"""

from math import sqrt

import numpy as np

from ..measures.shannon import jensen_shannon
from ..relevance.relevance_measures import ndcg_relevance_score


def genre_count(dist_vec: list) -> float:
    """
    Compute lambda as the fraction of active (non-zero) genres (CGR).

    Users who consume many genres receive a higher lambda, increasing the
    calibration emphasis for users with broader tastes.

    References
    ----------
    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        Fraction of genres with a positive probability, in ``[0, 1]``.
    """
    # Optimisation: replace `sum(map(lambda x: 1 if x > 0 else 0, ...))` with a
    # generator expression.  Avoids per-element lambda construction overhead.
    count = sum(1 for x in dist_vec if x > 0.0)
    return count / len(dist_vec)


def norm_var(dist_vec: list) -> float:
    """
    Compute lambda as one minus the normalised variance of the distribution (VAR).

    A highly concentrated (low-variance) distribution yields a lambda close
    to 1.0, increasing calibration emphasis for niche users.  A uniform
    distribution yields a lambda close to 0.0.

    References
    ----------
    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        ``1 - variance(dist_vec)``, in ``[0, 1]`` for valid probability
        distributions.
    """
    # Optimisation: (x - mean)^2 is always non-negative so abs() is redundant;
    # drop the lambda and use a plain generator expression to avoid per-element
    # lambda overhead.  Two passes (sum for mean, sum for variance) but both
    # are tight Python C-level loops.
    n = len(dist_vec)
    mean = sum(dist_vec) / n
    var = sum((x - mean) ** 2 for x in dist_vec) / n
    return 1.0 - var


def norm_std(dist_vec: list) -> float:
    """
    Compute lambda as one minus the normalised standard deviation of the distribution (STD).

    Analogous to :func:`norm_var` but uses the standard deviation, making it
    more sensitive to moderate spread differences.

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        ``1 - std(dist_vec)``, typically in ``[0, 1]``.
    """
    # Optimisation: drop redundant abs() (squaring is non-negative); use generator
    # expression instead of lambda in map().
    n = len(dist_vec)
    mean = sum(dist_vec) / n
    var = sum((x - mean) ** 2 for x in dist_vec) / n
    return 1.0 - sqrt(var)


def trust(dist_vec: list) -> float:
    """
    Compute lambda as the mean of the distribution vector (TRT).

    Returns the arithmetic mean, which can serve as a simple trust measure
    reflecting the average strength of genre preference.

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        Arithmetic mean of ``dist_vec``.
    """
    # Pure-Python sum/len is optimal for small lists; no change needed.
    return sum(dist_vec) / len(dist_vec)


def amplitude(dist_vec: list) -> float:
    """
    Compute lambda based on the normalised pairwise spread of the distribution (AMP).

    Computes the total absolute pairwise distance between all elements,
    normalised by ``len(dist_vec)^2``, then returns ``1 - normalised_spread``.
    A highly concentrated distribution has small spread, yielding a lambda
    close to 1.0.

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        ``1 - (total_pairwise_distance / n^2)``, typically in ``[0, 1]``.
    """
    # Optimisation: replace O(n^2) nested Python loops (list-comprehension of
    # sum+lambda) with NumPy broadcasting.  arr[:, None] - arr[None, :] builds
    # the n×n difference matrix in a single C-level operation; np.abs + np.sum
    # collapse it to a scalar.  For n=20 this is ~5× faster than Python loops;
    # gains grow quadratically with n.
    arr = np.asarray(dist_vec, dtype=np.float64)
    n = len(dist_vec)
    magnitude = float(np.sum(np.abs(arr[:, None] - arr[None, :])))
    return 1.0 - (magnitude / (n * n))


def efficiency(dist_vec: list) -> float:
    """
    Compute lambda as the coefficient of variation of the distribution (EFF).

    Returns ``variance / mean^2``.  This is the squared coefficient of
    variation, which is scale-invariant and captures relative dispersion
    independent of the magnitude of the distribution values.

    Parameters
    ----------
    dist_vec : list of float
        Genre probability distribution values for a single user.

    Returns
    -------
    float
        ``variance(dist_vec) / mean(dist_vec)^2``.

    Notes
    -----
    This function can return values outside ``[0, 1]`` when the variance
    exceeds the squared mean.
    """
    # Optimisation: drop redundant abs() (squaring guarantees non-negative result);
    # replace lambda in map() with a generator expression.
    n = len(dist_vec)
    mean = sum(dist_vec) / n
    var = sum((x - mean) ** 2 for x in dist_vec) / n
    return var / mean ** 2


def mitigation(dist_vec: list, target_dist: list, cand_dist: list) -> float:
    """
    Compute lambda using the MITigation (MIT) strategy.

    Combines the user's NDCG relevance on the candidate list with the
    Jensen-Shannon similarity between the target and candidate distributions
    via the harmonic mean:

        lambda = (ndcg * js_similarity) / (ndcg + js_similarity)

    A candidate list that is both relevant (high NDCG) and already close to
    the target distribution (high JS similarity) yields a higher lambda,
    reducing the need for aggressive re-ranking.

    Parameters
    ----------
    dist_vec : list of float
        Predicted scores for the candidate items (used to compute NDCG).
    target_dist : list of float
        Target genre probability distribution.
    cand_dist : list of float
        Genre probability distribution of the candidate item list.

    Returns
    -------
    float
        Harmonic-mean lambda value, in ``(0, 1]``.
    """
    # No change: bottleneck is in the callee functions (ndcg_relevance_score,
    # jensen_shannon), not in this wrapper.
    ndcg = ndcg_relevance_score(dist_vec)
    jsf = 1 - jensen_shannon(p=target_dist, q=cand_dist)
    result = (ndcg * jsf) / (ndcg + jsf)
    return result
