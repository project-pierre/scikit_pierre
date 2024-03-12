"""
This file contains all chi family equations.
"""
import math


def squared_euclidean(p: list, q: list) -> float:
    """
    Squared Euclidean (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum((p_i - q_i) ** 2 for p_i, q_i in zip(p, q))


def person_chi_square(p: list, q: list) -> float:
    """
    Pearson Chi-Square (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return ((p_i - q_i) ** 2) / q_i
        except ZeroDivisionError:
            return ((p_i - q_i) ** 2) / 0.00001
        except ArithmeticError:
            return 0.0

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def neyman_square(p: list, q: list) -> float:
    """
    Neyman Square (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (p_i - q_i) ** 2 / p_i
        except ZeroDivisionError:
            p_a = 0.00001 if p_i == 0 else p_i
            q_b = 0.00001 if q_i == 0 else q_i
            return (p_a - q_b) ** 2 / p_a
        except ArithmeticError:
            return 0.0

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def squared_chi_square(p: list, q: list) -> float:
    """
    Squared Chi Square (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (p_i - q_i) ** 2 / (p_i + q_i)
        except ZeroDivisionError:
            p_a = 0.00001 if p_i == 0 else p_i
            q_b = 0.00001 if q_i == 0 else q_i
            return (p_a - q_b) ** 2 / (p_a + q_b)
        except ArithmeticError:
            return 0.0

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def probabilistic_symmetric_chi_square(p: list, q: list) -> float:
    """
    Probabilistic Symmetric Chi Square (p, q) divergence.
    Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (p_i - q_i) ** 2 / (p_i + q_i)
        except ZeroDivisionError:
            p_a = 0.00001 if p_i == 0 else p_i
            q_b = 0.00001 if q_i == 0 else q_i
            return (p_a - q_b) ** 2 / (p_a + q_b)
        except ArithmeticError:
            return 0.0

    return 2 * sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def divergence(p: list, q: list) -> float:
    """
    Divergence (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (p_i - q_i) ** 2 / (p_i + q_i) ** 2
        except ZeroDivisionError:
            p_a = 0.00001 if p_i == 0 else p_i
            q_b = 0.00001 if q_i == 0 else q_i
            return (p_a - q_b) ** 2 / (p_a + q_b) ** 2
        except ArithmeticError:
            return 0.0

    return 2 * sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def clark(p: list, q: list) -> float:
    """
    Clark (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (abs(p_i - q_i) / (p_i + q_i)) ** 2
        except ZeroDivisionError:
            p_a = 0.00001 if p_i == 0 else p_i
            q_b = 0.00001 if q_i == 0 else q_i
            return (abs(p_a - q_b) / (p_a + q_b)) ** 2
        except ArithmeticError:
            return 0.0

    return math.sqrt(sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)]))


def additive_symmetric_chi_squared(p: list, q: list) -> float:
    """
    Additive Symmetric Chi Square (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        try:
            return (((p_i - q_i) ** 2) * (p_i + q_i)) / (p_i * q_i)
        except ZeroDivisionError:
            return (((p_i - q_i) ** 2) * (p_i + q_i)) / 0.00001
        except ArithmeticError:
            return 0.0

    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])
