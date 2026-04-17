"""
Class/genre-based probability distribution functions.

All functions accept a ``dict`` of :class:`~scikit_pierre.models.item.Item`
instances (keyed by item ID) and return a ``dict`` mapping genre/class label
to a non-negative float.  The ``*_P`` variants additionally normalise the
output so that values sum to 1.0.
"""


# ############################################################################################### #
# ######################################### Class Based ######################################### #
# ############################################################################################### #
def class_weighted_strategy(items: dict) -> dict:
    """
    Compute the Class Weighted Strategy (CWS) distribution.

    For each genre *g*, the distribution value is::

        CWS(g) = sum(score_i * genre_weight_ig) / sum(score_i)

    where the sums range over all items whose ``classes`` dict contains *g*.
    Genres where the numerator or denominator is non-positive receive a small
    epsilon (1e-5) instead of 0.0 to avoid issues with logarithm-based
    divergence measures downstream.

    References
    ----------
    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112
    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045
    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
        Each ``Item`` must have ``score`` and ``classes`` set.

    Returns
    -------
    dict
        Mapping of genre label -> CWS value.
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
    Compute the Weighted Probability Strategy (WPS) distribution.

    Normalises the CWS output so that genre values sum to 1.0, giving it
    proper probability semantics.

    References
    ----------
    - Silva and Durão (2022). https://arxiv.org/abs/2204.03706

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised genre probability distribution summing to 1.0.
    """
    distribution = class_weighted_strategy(items)
    total = sum(value for g, value in distribution.items())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


def pure_genre(items: dict) -> dict:
    """
    Compute the Pure Genre Distribution (PGD).

    Accumulates the raw genre weights from each item's ``classes`` dict
    without score weighting.  Useful as a simple genre-frequency baseline.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Mapping of genre label -> accumulated genre weight.
    """
    distribution = {}
    for _, item in items.items():
        for category, genre_value in item.classes.items():
            distribution[category] = distribution.get(category, 0.) + genre_value
    return distribution


def pure_genre_with_probability_property(items: dict) -> dict:
    """
    Compute the Pure Genre Distribution with Probability Property (PGD_P).

    Normalises the PGD output so that genre values sum to 1.0.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised genre probability distribution summing to 1.0.
    """
    dist = pure_genre(items)
    norm = sum(dist.values())
    distribution = {g: value / norm for g, value in dist.items()}
    return distribution

