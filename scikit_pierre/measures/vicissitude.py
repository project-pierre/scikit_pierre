"""
Vicissitude family of pairwise distribution divergence measures.

These are variants of the chi-square and wave-hedges measures that use the
*minimum* or *maximum* of the element pair as the denominator, making them
more robust to imbalanced distributions.

Zero values in *p* or *q* are replaced by a small epsilon (1e-5) to avoid
division-by-zero.

Reference
---------
Cha, S.-H. (2007). Comprehensive study of distance/similarity measures
between probability density functions.
https://www.gly.fsu.edu/~parker/geostats/Cha.pdf
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

    left = right = 0.0
    for p_i, q_i in zip(p, q):
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        diff_sq = (p_a - q_b) ** 2
        left += diff_sq / p_a
        right += diff_sq / q_b
    return max(left, right)


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

    left = right = 0.0
    for p_i, q_i in zip(p, q):
        p_a = 0.00001 if p_i == 0 else p_i
        q_b = 0.00001 if q_i == 0 else q_i
        diff_sq = (p_a - q_b) ** 2
        left += diff_sq / p_a
        right += diff_sq / q_b
    return min(left, right)
