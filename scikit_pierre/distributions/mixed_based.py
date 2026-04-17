"""
Mixed probability distribution functions (entropy + time weighting).

Combines the GLEB entropy weighting with the TWB timestamp weighting to
produce distributions that simultaneously capture recency and information
content of each genre.

All public functions accept a ``dict`` of
:class:`~scikit_pierre.models.item.Item` instances and return a ``dict``
mapping genre label to a float.
"""
from collections import Counter

from math import log2


def mixed_gleb_twb(items: dict) -> dict:
    """
    Compute the mixed GLEB + TWB distribution (TWB_GLEB).

    Combines global entropy weighting with temporal (timestamp) weighting:
    the entropy term discounts overrepresented genres while the time weight
    favours recent interactions.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
        Each ``Item`` must have ``score``, ``time``, and ``classes`` set.

    Returns
    -------
    dict
        Mapping of genre label -> TWB_GLEB distribution value.
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
    Compute the TWB_GLEB distribution normalised to sum to 1.0 (TWB_GLEB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised TWB_GLEB genre probability distribution summing to 1.0.
    """
    distribution = mixed_gleb_twb(items)
    total = sum(distribution.values())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution

