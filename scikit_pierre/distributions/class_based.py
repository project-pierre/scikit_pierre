"""
This file presents the distributions based on the class/genre.
"""


# ############################################################################################### #
# ######################################### Class Based ######################################### #
# ############################################################################################### #
def class_weighted_strategy(items: dict) -> dict:
    """
    The Class Weighted Strategy - (CWS). The reference for this implementation are from:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045

    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute() -> None:
        for _, item in items.items():
            for category, genre_value in item.classes.items():
                numerator[category] = numerator.get(category, 0) + item.score * genre_value
                denominator[category] = denominator.get(category, 0) + item.score

    def genre(g: str) -> float:
        if ((g in denominator and denominator[g] > 0.0) and
                (g in numerator and numerator[g] > 0.0)):
            return numerator[g] / denominator[g]
        return 0.00001

    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def weighted_probability_strategy(items: dict) -> dict:
    """
    The Weighted Probability Strategy - (WPS). The reference for this implementation are from:

    - Silva and Durão (2022). https://arxiv.org/abs/2204.03706

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    distribution = class_weighted_strategy(items)
    total = sum(value for g, value in distribution.items())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


def pure_genre(items: dict) -> dict:
    """
    The Pure Genre Distribution - (PGD). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    distribution = {}
    for _, item in items.items():
        for category, genre_value in item.classes.items():
            distribution[category] = distribution.get(category, 0.) + genre_value
    return distribution


def pure_genre_with_probability_property(items: dict) -> dict:
    """
    The Pure Genre Distribution with Probability Property - (PGD_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    dist = pure_genre(items)
    norm = sum(dist.values())
    distribution = {g: value / norm for g, value in dist.items()}
    return distribution

# ############################################################################################### #
# ######################################### Unrevised ########################################## #
# ############################################################################################### #
