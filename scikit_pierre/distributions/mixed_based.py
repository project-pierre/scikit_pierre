"""
This file contains the distribution functions based on mixed equations.
"""
from collections import Counter

from math import log2


def mixed_gleb_twb(items: dict) -> dict:
    """
    The Global and Local Entropy Based with Time Weight Based - (TWB_GLEB).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def global_entropy() -> dict:
        genre_list = [
            category
            for index, item in items.items()
            for category, genre_value in item.classes.items()
        ]

        n = len(genre_list)
        total = dict(Counter(genre_list))
        return {t: total[t] / n for t in dict(total)}

    def compute() -> None:
        for _, item in items.items():
            for category, genre_value in item.classes.items():
                ent = -(genre_global[category] * genre_value) * log2(
                    genre_global[category] * genre_value)
                numerator[category] = numerator.get(category, 0) + item.score * item.time * ent
                denominator[category] = denominator.get(category, 0) + item.score

    def genre(g: str) -> float:
        if (g in denominator and denominator[g] > 0.0) and (
                g in numerator and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        return 0.00001

    genre_global = global_entropy()
    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def mixed_gleb_twb_with_probability_property(items: dict) -> dict:
    """
    The Global and Local Entropy Based with Time Window Based with Probability Property -
    (TWB_GLEB_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    distribution = mixed_gleb_twb(items)
    total = sum(distribution.values())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution

# ############################################################################################### #
# ######################################### Unrevised ########################################## #
# ############################################################################################### #
