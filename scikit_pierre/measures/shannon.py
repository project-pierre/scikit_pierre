"""
This file contains all shanon family equations.
"""

from math import log


def kullback_leibler(p: list, q: list) -> float:
    """
    Kullback-Leibler (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return p_a * log(p_a / q_b)

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def jeffreys(p: list, q: list) -> float:
    """
    Jeffreys (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return (p_a - q_b) * log(p_a / q_b)

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def k_divergence(p: list, q: list) -> float:
    """
    K Divergence (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return p_a * log((2 * p_a) / (p_a + q_b))

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def topsoe(p: list, q: list) -> float:
    """
    Topsoe (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return (p_a * log((2 * p_a) / (p_a + q_b))) + (q_b * log((2 * q_b) / (p_a + q_b)))

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def jensen_shannon(p: list, q: list) -> float:
    """
    Jensen Shannon (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute_left(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return p_a * log((2 * p_a) / (p_a + q_b))

    def compute_right(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return q_b * log((2 * q_b) / (p_a + q_b))

    return (1 / 2) * (sum(compute_left(p_i, q_i) for p_i, q_i in zip(p, q)) +
                      sum(compute_right(p_i, q_i) for p_i, q_i in zip(p, q)))


def jensen_difference(p: list, q: list) -> float:
    """
    Jensen Difference (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return (((p_a * log(p_a)) + (q_b * log(q_b))) / 2) - (
                    ((p_a + q_b) / 2) * log((p_a + q_b) / 2))

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))
