"""
Factory accessor for relevance scoring functions.

Provides a single entry-point that maps a string key to the corresponding
callable from :mod:`relevance_measures`.
"""

from . import relevance_measures


def relevance_measures_funcs(relevance: str = "SUM"):
    """
    Return the relevance scoring function identified by *relevance*.

    Parameters
    ----------
    relevance : str, optional
        Acronym for the desired relevance function.  Supported values:

        - ``"SUM"``    — plain summation (:func:`sum_relevance_score`)
        - ``"NDCG"``   — Normalized DCG (:func:`ndcg_relevance_score`)
        - ``"UREL"``   — utility-based (:func:`utility_relevance_scores`)
        - ``"TECREC"`` — TecRec discount (:func:`relevance_tecrec`)

    Returns
    -------
    callable
        A function with signature ``(scores: list) -> float``.

    Raises
    ------
    NameError
        If *relevance* does not match any known key.
    """
    if relevance == "SUM":
        return relevance_measures.sum_relevance_score
    if relevance == "NDCG":
        return relevance_measures.ndcg_relevance_score
    if relevance == "UREL":
        return relevance_measures.utility_relevance_scores
    if relevance == "TECREC":
        return relevance_measures.relevance_tecrec
    raise NameError(f"Relevance Measure not found! {relevance}")
