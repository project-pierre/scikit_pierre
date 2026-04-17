"""
Entropy-based probability distribution functions.

These distributions weight each genre by an entropy term that captures
how globally and locally concentrated the genre is across the item set,
promoting less common genres relative to dominant ones.

All public functions accept a ``dict`` of
:class:`~scikit_pierre.models.item.Item` instances and return a
``dict`` mapping genre label to a float.
"""
from collections import Counter

from math import log2


def global_local_entropy_based(items: dict) -> dict:
    """
    Compute the Global and Local Entropy Based (GLEB) distribution.

    First computes a global entropy weight for each genre based on its
    frequency across all items; then weights each item's genre contribution
    by both the interaction score and this entropy weight.  This discounts
    over-represented genres and boosts underrepresented ones.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
        Each ``Item`` must have ``score`` and ``classes`` set.

    Returns
    -------
    dict
        Mapping of genre label -> GLEB distribution value.
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

    def compute():
        for _, item in items.items():
            for category, genre_value in item.classes.items():
                log_entropy = log2(genre_global[category] * genre_value)
                ent = -(genre_global[category] * genre_value) * log_entropy
                numerator[category] = numerator.get(category, 0) + item.score * ent
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


def global_local_entropy_based_with_probability_property(items: dict) -> dict:
    """
    Compute the GLEB distribution normalised to sum to 1.0 (GLEB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised GLEB genre probability distribution summing to 1.0.
    """
    distribution = global_local_entropy_based(items)
    total = sum(distribution.values())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution

