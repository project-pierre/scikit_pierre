"""
Fidelity family of pairwise distribution similarity and divergence measures.

Functions that compute *similarity* return higher values for more similar
distributions (maximum 1.0 for identical distributions); functions that
compute *divergence* return lower values for closer distributions (0.0
for identical distributions).

All functions accept two equal-length lists *p* and *q* of non-negative
floats.

Reference
---------
Cha, S.-H. (2007). Comprehensive study of distance/similarity measures
between probability density functions.
https://www.gly.fsu.edu/~parker/geostats/Cha.pdf
"""

import math


def fidelity(p: list, q: list) -> float:
    """
    Compute the Fidelity (Bhattacharyya coefficient) similarity between p and q.

    Higher values indicate more similar distributions; identical distributions
    yield 1.0.

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Fidelity value in ``[0, 1]``.
    """

    return sum(math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q))


def bhattacharyya(p: list, q: list) -> float:
    """
    Compute the Bhattacharyya divergence between p and q.

    Defined as ``-ln(fidelity(p, q))``.  A value of 0.0 means the
    distributions are identical; larger values indicate greater divergence.
    When the fidelity is zero (completely disjoint distributions) a small
    epsilon (1e-5) is used before taking the logarithm.

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Bhattacharyya distance in ``[0, +inf)``.
    """

    def compute() -> float:
        value = sum(math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q))
        if value == 0:
            return 0.00001
        return value

    return -math.log(compute())


def hellinger(p: list, q: list) -> float:
    """
    Compute the Hellinger distance between p and q.

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Hellinger distance in ``[0, 2]`` (equals 0 for identical
        distributions, equals 2 for completely disjoint distributions).
    """
    return math.sqrt(2 * sum((math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)))


def matusita(p: list, q: list) -> float:
    """
    Compute the Matusita distance between p and q.

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Matusita distance in ``[0, sqrt(2)]``.
    """

    def compute(p_i: float, q_i: float) -> float:
        return (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

    return math.sqrt(sum(compute(p_i, q_i) for p_i, q_i in zip(p, q)))


def squared_chord_similarity(p: list, q: list) -> float:
    """
    Compute the Squared-chord similarity between p and q.

    Higher values indicate more similar distributions.

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Squared-chord similarity in ``[-1, 1]``.
    """
    return 2 * sum(math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q)) - 1


def squared_chord_divergence(p: list, q: list) -> float:
    """
    Compute the Squared-chord divergence between p and q.

    Equivalent to the squared Matusita distance (without the outer square root).

    Parameters
    ----------
    p : list of float
        First probability distribution.  Must be the same length as *q*.
    q : list of float
        Second probability distribution.  Must be the same length as *p*.

    Returns
    -------
    float
        Squared-chord divergence in ``[0, 2]``.
    """
    return sum((math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q))
