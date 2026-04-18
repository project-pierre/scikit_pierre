"""
Timestamp-weighted probability distribution functions.

These distributions incorporate the normalised interaction timestamp
(``item.time``) so that more recent interactions contribute more strongly
to the genre distribution.

All public functions accept a ``dict`` of
:class:`~scikit_pierre.models.item.Item` instances and return a ``dict``
mapping genre label to a float.
"""


# ############################################################################################### #
# ######################################### Time Based ########################################## #
# ############################################################################################### #


def time_weighted_based(items: dict) -> dict:
    """
    Compute the Time Weight Based (TWB) distribution.

    Extends CWS by multiplying each item's score by its normalised timestamp
    (``item.time``) before accumulating genre weights, so that recent
    interactions have a stronger influence.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
        Each ``Item`` must have ``score``, ``time``, and ``classes`` set.

    Returns
    -------
    dict
        Mapping of genre label -> TWB distribution value.
    """
    numerator = {}
    denominator = {}

    def compute() -> None:
        for _, item in items.items():
            for category, genre_value in item.classes.items():
                numerator[category] = numerator.get(category,
                                                    0) + item.time * item.score * genre_value
                denominator[category] = denominator.get(category, 0) + item.score

    def genre(g: str) -> float:
        if (g in denominator and denominator[g] > 0.0) and (
                g in numerator and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        return 0.00001

    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def time_weighted_based_with_probability_property(items: dict) -> dict:
    """
    Compute the TWB distribution normalised to sum to 1.0 (TWB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised TWB genre probability distribution summing to 1.0.
    """
    distribution = time_weighted_based(items)
    total = sum(value for g, value in distribution.items())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


def time_genre(items: dict) -> dict:
    """
    Compute the Time Genre Distribution (TGD).

    Weights genre contributions solely by the normalised timestamp
    (``item.time``), ignoring interaction scores.  This focuses on the
    temporal recency of genre exposure rather than rating magnitude.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
        Each ``Item`` must have ``time`` and ``classes`` set.

    Returns
    -------
    dict
        Mapping of genre label -> TGD distribution value.
    """
    numerator = {}
    denominator = {}

    def compute() -> None:
        for _, item in items.items():
            for category, genre_value in item.classes.items():
                numerator[category] = numerator.get(category, 0.) + item.time * genre_value
                denominator[category] = denominator.get(category, 0.) + item.time

    def genre(g: str) -> float:
        if (g in denominator and denominator[g] > 0.0) and (
                g in numerator and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        return 0.00001

    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def time_genre_with_probability_property(items: dict) -> dict:
    """
    Compute the TGD distribution normalised to sum to 1.0 (TGD_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.

    Returns
    -------
    dict
        Normalised TGD genre probability distribution summing to 1.0.
    """
    dist = time_genre(items)
    norm = sum(dist.values())
    distribution = {g: value / norm for g, value in dist.items()}
    return distribution
