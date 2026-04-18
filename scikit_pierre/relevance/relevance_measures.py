"""
Relevance scoring functions for ranked recommendation lists.

Each function receives a list of per-item scores (one per position) and
returns a single scalar that summarises how relevant the list is.  These
scalars are used as the relevance component inside the calibration
trade-off objective.
"""

from math import log2

import numpy as np


def sum_relevance_score(scores: list) -> float:
    """
    Compute list relevance as the plain sum of per-item scores.

    References
    ----------
    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112
    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    Parameters
    ----------
    scores : list of float
        Per-item relevance weights in ranked order.

    Returns
    -------
    float
        Sum of all values in ``scores``.
    """
    return sum(scores)


def ndcg_relevance_score(scores: list) -> float:
    """
    Compute list relevance using Normalized Discounted Cumulative Gain (NDCG).

    The ideal DCG is computed from the same score list sorted in descending
    order, so the result is always in [0, 1].  Returns 0.0 for an empty or
    all-zero list.

    References
    ----------
    - Silva and Durão (2022).
      https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4195978

    Parameters
    ----------
    scores : list of float
        Per-item relevance scores in ranked order.

    Returns
    -------
    float
        NDCG value in ``[0, 1]``.
    """

    def __dcg_at_k(rel: list) -> float:
        return sum((2 ** score - 1) / log2(ix + 2) for ix, score in enumerate(rel))

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
    Compute list relevance using a logarithmic utility function.

    Utility is defined as ``sum(score_i * (1 / log(i+1)))`` starting from
    position 1.  The result is normalised by the ideal (sorted-descending)
    utility, so the output is in ``[0, 1]``.  Returns 0.0 when the ideal
    utility is zero.

    References
    ----------
    - Sacharidis, Mouratidis, Kleftogiannis (2019).
      https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7946&context=sis_research

    Parameters
    ----------
    scores : list of float
        Per-item relevance scores in ranked order.

    Returns
    -------
    float
        Normalised utility value in ``[0, 1]``.
    """

    n = len(scores)
    weights = 1.0 / np.log(np.arange(2, n + 2))  # 1/log(1)…1/log(n), 1-indexed positions
    arr = np.asarray(scores, dtype=float)
    summation = float(np.dot(arr, weights))
    ideal = float(np.dot(np.sort(arr)[::-1], weights))

    if not ideal:
        return 0.0
    return summation / ideal

def relevance_tecrec(scores: list) -> float:
    """
    Compute list relevance using the TecRec position-discounted summation.

    Each score is discounted by ``1 / (|scores| + 1)`` regardless of
    position.  This is a simplified uniform-discount variant.

    References
    ----------
    - Xing Zhao, Ziwei Zhu, James Caverlee (2021).
      https://dl.acm.org/doi/abs/10.1145/3442381.3450099

    Parameters
    ----------
    scores : list of float
        Per-item relevance scores in ranked order.

    Returns
    -------
    float
        Position-discounted sum of scores.
    """
    return sum(scores) / (len(scores) + 1)
