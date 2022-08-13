from math import sqrt


def minkowski(p: list, q: list, d: int = 3, **kwargs) -> float:
    """
    Minkowski (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param d: A value to indicate the Minkowski formulation, 1 is City Block, 2 is Euclidean, etc.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum([abs(p_i - q_j) ** d for p_i, q_j in zip(p, q)]) ** (1 / d)


def euclidean(p: list, q: list, **kwargs) -> float:
    """
    Euclidean (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sqrt(sum([abs(p_i - q_j) ** 2 for p_i, q_j in zip(p, q)]))


def city_block(p: list, q: list, **kwargs) -> float:
    """
    City Block (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum([abs(p_i - q_j) for p_i, q_j in zip(p, q)])


def chebyshev(p: list, q: list, **kwargs) -> float:
    """
    Chebyshev (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return max([abs(p_i - q_j) for p_i, q_j in zip(p, q)])
