import math


def fidelity(p: list, q: list, **kwargs) -> float:
    """
    Fidelity (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    return sum([math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q)])


def bhattacharyya(p: list, q: list, **kwargs) -> float:
    """
    Bhattacharyya (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute() -> float:
        value = sum([math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q)])
        if value == 0:
            return 0.00001
        return value

    return -math.log(compute())


def hellinger(p: list, q: list, **kwargs) -> float:
    """
    Hellinger (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return math.sqrt(2 * sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]))


def matusita(p: list, q: list, **kwargs) -> float:
    """
    Matusita (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """

    def compute(p_i: float, q_i: float) -> float:
        return (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

    return math.sqrt(sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)]))


def squared_chord_similarity(p: list, q: list, **kwargs) -> float:
    """
    Squared-chord (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return 2 * sum([math.sqrt(p_i * q_i) for p_i, q_i in zip(p, q)]) - 1


def squared_chord_divergence(p: list, q: list, **kwargs) -> float:
    """
    Squared-chord (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)])
