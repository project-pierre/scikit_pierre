from . import relevance_measures


def relevance_measures_funcs(relevance: str = "SUM"):
    """
    :param relevance: A string that's represents the relevance function name.

    :return: A relevance function.
    """
    if relevance == "SUM":
        return relevance_measures.sum_relevance_score
    elif relevance == "NDCG":
        return relevance_measures.ndcg_relevance_score
    else:
        raise Exception("Relevance measure not found!")
