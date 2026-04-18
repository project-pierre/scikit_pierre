"""
Minkowski family of pairwise distance measures.

All functions accept two equal-length lists *p* and *q* of floats and
return a non-negative float (divergence convention: lower = closer).

Reference
---------
Cha, S.-H. (2007). Comprehensive study of distance/similarity measures
between probability density functions.
https://www.gly.fsu.edu/~parker/geostats/Cha.pdf
"""
from math import sqrt


def minkowski(p: list, q: list, d: int = 3) -> float:
    """
    Compute the Minkowski distance of order *d* between p and q.

    Parameters
    ----------
    p : list of float
        First distribution vector.  Must be the same length as *q*.
    q : list of float
        Second distribution vector.  Must be the same length as *p*.
    d : int, optional
        Order of the Minkowski metric.  ``d=1`` is the City-Block (Manhattan)
        distance; ``d=2`` is the Euclidean distance.  Defaults to 3.

    Returns
    -------
    float
        Minkowski distance in ``[0, +inf)``.
    """
    return sum(abs(p_i - q_j) ** d for p_i, q_j in zip(p, q)) ** (1 / d)


def euclidean(p: list, q: list) -> float:
    """
    Compute the Euclidean distance between p and q.

    Equivalent to :func:`minkowski` with ``d=2``.

    Parameters
    ----------
    p : list of float
        First distribution vector.  Must be the same length as *q*.
    q : list of float
        Second distribution vector.  Must be the same length as *p*.

    Returns
    -------
    float
        Euclidean distance in ``[0, +inf)``.
    """
    return sqrt(sum(abs(p_i - q_j) ** 2 for p_i, q_j in zip(p, q)))


def city_block(p: list, q: list) -> float:
    """
    Compute the City-Block (Manhattan / L1) distance between p and q.

    Equivalent to :func:`minkowski` with ``d=1``.

    Parameters
    ----------
    p : list of float
        First distribution vector.  Must be the same length as *q*.
    q : list of float
        Second distribution vector.  Must be the same length as *p*.

    Returns
    -------
    float
        City-Block distance in ``[0, +inf)``.
    """
    return sum(abs(p_i - q_j) for p_i, q_j in zip(p, q))


def chebyshev(p: list, q: list) -> float:
    """
    Compute the Chebyshev (L-infinity) distance between p and q.

    Returns the maximum absolute element-wise difference.

    Parameters
    ----------
    p : list of float
        First distribution vector.  Must be the same length as *q*.
    q : list of float
        Second distribution vector.  Must be the same length as *p*.

    Returns
    -------
    float
        Chebyshev distance in ``[0, +inf)``.
    """
    return max(abs(p_i - q_j) for p_i, q_j in zip(p, q))
