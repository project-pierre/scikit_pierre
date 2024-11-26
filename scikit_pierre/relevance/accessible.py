"""
This file allow to access all relevance measures.
"""

from . import relevance_measures


def relevance_measures_funcs(relevance: str = "SUM"):
    """
    :param relevance: A string that's represents the relevance function name.

    :return: A relevance function.
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
