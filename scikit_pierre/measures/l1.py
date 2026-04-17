"""
L1 family of pairwise distribution divergence measures.

All functions accept two equal-length lists *p* and *q* of non-negative
floats and return a non-negative float (divergence convention: lower
values indicate closer distributions).

Division-by-zero conditions are handled by substituting a small epsilon
(1e-5) as the denominator.

Reference
---------
Cha, S.-H. (2007). Comprehensive study of distance/similarity measures
between probability density functions.
https://www.gly.fsu.edu/~parker/geostats/Cha.pdf
"""
import math


def sorensen(p: list, q: list) -> float:
    """
    Sorensen (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))
    denominator = sum(p_i + q_i for p_i, q_i in zip(p, q))
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def gower(p: list, q: list) -> float:
    """
    Gower (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return (1 / len(p)) * sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))


def soergel(p: list, q: list) -> float:
    """
    Soergel (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))
    denominator = sum(max([p_i, q_i]) for p_i, q_i in zip(p, q))
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def kulczynski_d(p: list, q: list) -> float:
    """
    Kulczynski d (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))
    denominator = sum(min([p_i, q_i]) for p_i, q_i in zip(p, q))
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def canberra(p: list, q: list) -> float:
    """
    Canberra (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        numerator = abs(p_i - q_i)
        denominator = p_i + q_i
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return numerator / 0.00001

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def lorentzian(p: list, q: list) -> float:
    """
    Lorentzian (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        smooth = 1 + abs(p_i - q_i)
        return math.log(smooth)

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))
