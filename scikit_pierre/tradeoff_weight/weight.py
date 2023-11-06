def genre_count(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:
    
    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    count = sum(map(lambda x: 1 if (x > .0) else 0, dist_vec))
    return count / len(dist_vec)


def norm_var(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return 1 - var


def norm_std(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:

    - xxxxxxxxx.

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    from math import sqrt
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return 1 - sqrt(var)


def trust(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:

    - xxxxxxxxx.

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    return sum(dist_vec) / len(dist_vec)


def amplitude(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:

    - xxxxxxxxx.

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    magnitude = sum([
        sum(
            map(lambda x: abs(x - y), dist_vec)) for y in dist_vec
        ])
    return 1 - (magnitude / len(dist_vec) ** 2)


def efficiency(dist_vec: list) -> float:
    """
    A function to compute the tradeoff lambda (weight).
    The reference for this implementation is from:

    - xxxxxxxxx.

    :param dist_vec: A list composed of float numbers.
    :return: A float between [0;1], which represent the degree of user genre preference.
    """
    mean = sum(dist_vec) / len(dist_vec)
    numerator = sum(map(lambda x: abs(x - mean) ** 2, dist_vec))
    var = numerator / len(dist_vec)
    return var / mean**2