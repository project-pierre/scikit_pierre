"""
This file contains all inner product family equations.
"""

import math


def inner_product(p: list, q: list, **kwargs) -> float:
    """
    Inner Product (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum([p_i * q_i for p_i, q_i in zip(p, q)])


def harmonic_mean(p: list, q: list, **kwargs) -> float:
    """
    Harmonic mean (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        numerator = p_i * q_i
        denominator = p_i + q_i
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return numerator / 0.00001

    return 2 * sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def cosine(p: list, q: list, **kwargs) -> float:
    """
    Cosine (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([p_i * q_i for p_i, q_i in zip(p, q)])
    p_denominator = math.sqrt(sum([p_i ** 2 for p_i in p]))
    q_denominator = math.sqrt(sum([q_i ** 2 for p_i, q_i in zip(p, q)]))
    denominator = p_denominator * q_denominator
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def kumar_hassebrook(p: list, q: list, **kwargs) -> float:
    """
    Kumar-Hassebrook (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([p_i * q_i for p_i, q_i in zip(p, q)])
    p_denominator = sum([p_i ** 2 for p_i in p])
    q_denominator = sum([q_i ** 2 for p_i, q_i in zip(p, q)])
    denominator = p_denominator + q_denominator - numerator
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def jaccard(p: list, q: list, **kwargs) -> float:
    """
    Jaccard (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([(p_i - q_i) ** 2 for p_i, q_i in zip(p, q)])
    p_denominator = sum([p_i ** 2 for p_i in p])
    q_denominator = sum([q_i ** 2 for p_i, q_i in zip(p, q)])
    prod_denominator = sum([p_i * q_i for p_i, q_i in zip(p, q)])
    denominator = p_denominator + q_denominator - prod_denominator
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def dice_similarity(p: list, q: list, **kwargs) -> float:
    """
    Dice (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = 2 * sum([p_i * q_i for p_i, q_i in zip(p, q)])
    denominator = sum([p_i ** 2 for p_i in p]) + sum([q_i ** 2 for q_i in q])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def dice_divergence(p: list, q: list, **kwargs) -> float:
    """
    Dice (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([(p_i - q_i) ** 2 for p_i, q_i in zip(p, q)])
    denominator = sum([p_i ** 2 for p_i in p]) + sum([q_i ** 2 for q_i in q])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001
