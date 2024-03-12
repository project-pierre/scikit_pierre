"""
This file contains all vicissitude family equations.
"""


def vicis_wave_hedges(p: list, q: list) -> float:
    """
    Vicis-Wave Hedges (p, q) divergence. Low values means close, high values means far.

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
        numerator = abs(p_a - q_b)
        denominator = min(p_a, q_b)
        return numerator / denominator

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def vicis_symmetric_chi_square(p: list, q: list) -> float:
    """
    Vicis-Symmetric Chi Square (p, q) divergence. Low values means close, high values means far.

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
        numerator = (p_a - q_b) ** 2
        denominator = min(p_a, q_b) ** 2
        return numerator / denominator

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def vicis_symmetric_chi_square_emanon3(p: list, q: list) -> float:
    """
    Vicis-Symmetric Chi Square Emanon 3(p, q) divergence.
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
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        numerator = (p_a - q_b) ** 2
        denominator = min(p_a, q_b)
        return numerator / denominator

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def vicis_symmetric_chi_square_emanon4(p: list, q: list) -> float:
    """
    Vicis-Symmetric Chi Square Emanon 4 (p, q) divergence.
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
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        numerator = (p_a - q_b) ** 2
        denominator = max(p_a, q_b)
        return numerator / denominator

    return sum(compute(p_i, q_i) for p_i, q_i in zip(p, q))


def max_symmetric_chi_square_emanon5(p: list, q: list) -> float:
    """
    Vicis-Symmetric Chi Square Emanon 5 (p, q) divergence.
    Low values means close, high values means far.

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
        numerator = (p_a - q_b) ** 2
        denominator = p_a
        return numerator / denominator

    def compute_right(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        numerator = (p_a - q_b) ** 2
        denominator = q_b
        return numerator / denominator

    return max([sum(compute_left(p_i, q_i) for p_i, q_i in zip(p, q)),
                sum(compute_right(p_i, q_i) for p_i, q_i in zip(p, q))])


def min_symmetric_chi_square_emanon6(p: list, q: list) -> float:
    """
    Vicis-Symmetric Chi Square Emanon 6 (p, q) divergence.
    Low values means close, high values means far.

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
        numerator = (p_a - q_b) ** 2
        denominator = p_a
        return numerator / denominator

    def compute_right(p_i: float, q_i: float) -> float:
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        numerator = (p_a - q_b) ** 2
        denominator = q_b
        return numerator / denominator

    return min([sum(compute_left(p_i, q_i) for p_i, q_i in zip(p, q)),
                sum(compute_right(p_i, q_i) for p_i, q_i in zip(p, q))])
