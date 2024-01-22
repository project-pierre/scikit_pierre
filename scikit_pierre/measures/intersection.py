def intersection_similarity(p: list, q: list, **kwargs) -> float:
    """
    Intersection (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return sum([min([p_i, q_i]) for p_i, q_i in zip(p, q)])


def intersection_divergence(p: list, q: list, **kwargs) -> float:
    """
    Intersection (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    return (1 / 2) * sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])


def wave_hedges(p: list, q: list, **kwargs) -> float:
    """
    Wave Hedges (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    def compute(p_i: float, q_i: float) -> float:
        numerator = abs(p_i - q_i)
        denominator = max([p_i, q_i])
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return numerator / 0.00001
    return sum([compute(p_i, q_i) for p_i, q_i in zip(p, q)])


def czekanowski_similarity(p: list, q: list, **kwargs) -> float:
    """
    Czekanowski (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = 2 * sum([min([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([p_i + q_i for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def czekanowski_divergence(p: list, q: list, **kwargs) -> float:
    """
    Czekanowski (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
    denominator = sum([p_i + q_i for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def motyka_similarity(p: list, q: list, **kwargs) -> float:
    """
    Motyka (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([min([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([p_i + q_i for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def motyka_divergence(p: list, q: list, **kwargs) -> float:
    """
    Motyka (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([max([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([p_i + q_i for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def kulczynski_s(p: list, q: list, **kwargs) -> float:
    """
    Kulczynski s (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([min([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def ruzicka(p: list, q: list, **kwargs) -> float:
    """
    Ruzicka (p, q) similarity. Low values means different, high values means similar.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([min([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([max([p_i, q_i]) for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001


def tanimoto(p: list, q: list, **kwargs) -> float:
    """
    Tanimoto (p, q) divergence. Low values means close, high values means far.

    The reference for this implementation are from:

    - CHA, S.-H (2007). "https://www.gly.fsu.edu/∼parker/geostats/Cha.pdf"

    :param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
    :return: A float between [0;+inf], which represent the distance between p and q.
    """
    numerator = sum([max([p_i, q_i]) - min([p_i, q_i]) for p_i, q_i in zip(p, q)])
    denominator = sum([max([p_i, q_i]) for p_i, q_i in zip(p, q)])
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return numerator / 0.00001
