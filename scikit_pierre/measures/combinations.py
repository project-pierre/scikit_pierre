"""
This file contains all combinations family equations.
"""
from math import log, sqrt


def taneja(p: list, q: list, **kwargs) -> float:
    """
    Taneja (p, q) divergence. Low values means close, high values means far.

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
        return ((p_a + q_b) / 2) * log((p_a + q_b) / (2 * sqrt(p_a * q_b)))

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def kumar_johnson(p: list, q: list, **kwargs) -> float:
    """
    Kumar Johnson (p, q) divergence. Low values means close, high values means far.

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
        return ((p_a ** 2 - q_b ** 2) ** 2) / (2 * (p_a * q_b) ** (3 / 2))

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def avg(p: list, q: list, **kwargs) -> float:
    """
    AVG (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float, diff: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return abs(p_a - q_b) + diff

    difference = max([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
    return sum([compute(p_i, q_i, difference) for p_i, q_i in zip(p, q)]) / 2


def weighted_total_variation(p: list, q: list, **kwargs) -> float:
    """
    Weighted Total Variation(P, Q) (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - Seymen et al. (2021). https://doi.org/10.1145/3460231.3478857

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        return (p_a + 1) * abs(p_a - q_b)

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)]) / 2
