"""
This file contains all relevance measure equations.
"""

from math import log
import numpy as np


def sum_relevance_score(scores: list) -> float:
    """
    Sum Revelance Score computes the list relevance.

    The reference for this implementation are from:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    :param scores: A list with float numbers,
        which represents the weight for each item in its position.

    :return: A float, which represent list weight.
    """
    return sum(scores)


def ndcg_relevance_score(scores: list) -> float:
    """
    The Normalized Discount Cumulative Gain (NDCG) computes the list relevance.

    The reference for this implementation are from:

    - Silva and Durão. (2022).

    :param scores: A list of float in which represents the relevance score
        for each item in its position.

    :return: A float which represents the list relevance.
    """

    def __dcg_at_k(rel: list) -> float:
        return sum(((2 ** score) - 1) / (np.log2(ix + 2)) for ix, score in enumerate(rel))

    def ndcg_at_k(rel: list) -> float:
        idcg = __dcg_at_k(sorted(rel, reverse=True))
        if not idcg:
            return 0.0
        return __dcg_at_k(rel) / idcg

    if scores is None or len(scores) < 1:
        return 0.0
    return ndcg_at_k(rel=scores)


def utility_relevance_scores(scores: list) -> float:
    """
    The ... computes the list relevance.

    The reference for this implementation are from:

    - Sacharidis, Mouratidis, Kleftogiannis (2019) -
        https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7946&context=sis_research

    :param scores: A list of float in which represents the relevance score
        for each item in its position.

    :return: A float which represents the list relevance.
    """

    def utility(values: list) -> float:
        return sum(ix * (1 / log(ix + 1)) for ix, value in enumerate(values, start=1))

    summation = utility(values=scores)
    ideal = utility(sorted(scores, reverse=True))

    return summation / ideal
