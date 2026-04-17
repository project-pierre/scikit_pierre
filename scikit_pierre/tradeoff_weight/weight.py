"""
Trade-off weight (lambda) functions.

These functions compute a scalar lambda in ``[0, 1]`` from a user's
genre-probability distribution vector.  Lambda controls the balance between
relevance and fairness (calibration) in the trade-off objective:

    utility = (1 - lambda) * relevance ± lambda * fairness

Higher lambda values emphasise calibration; lower values emphasise relevance.
"""

from math import sqrt

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
    count = sum(map(lambda x: 1 if (x > .0) else 0, dist_vec))
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
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return 1 - var


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
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return 1 - sqrt(var)


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
    summation = [sum(map(lambda x: abs(x - y), dist_vec)) for y in dist_vec]
    magnitude = sum(summation)
    return 1 - (magnitude / len(dist_vec) ** 2)


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
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return var / mean**2


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
    ndcg = ndcg_relevance_score(dist_vec)
    jsf = 1 - jensen_shannon(p=target_dist, q=cand_dist)
    result = (ndcg * jsf) / (ndcg + jsf)
    return result
