"""
Utilities for computing per-user genre/class probability distributions.

Provides DataFrame and dict variants of the same computation, plus a
helper for aligning two genre-keyed distribution dicts into paired
numeric vectors suitable for divergence measure functions.
"""
from pandas import DataFrame, concat

from .accessible import distributions_funcs
from ..models.item import ItemsInMemory


def computer_users_distribution_pandas(
        users_preference_set: DataFrame, items_df: DataFrame, distribution: str
) -> DataFrame:
    """
    Compute per-user genre distributions and return the result as a DataFrame.

    Parameters
    ----------
    users_preference_set : DataFrame
        User interaction data.  Expected columns:
        ``USER_ID``, ``ITEM_ID``, ``TRANSACTION_VALUE``, and optionally
        ``TIMESTAMP``.
    items_df : DataFrame
        Item catalogue with at least ``ITEM_ID`` and ``GENRES`` columns.
    distribution : str
        Acronym identifying the distribution strategy (see
        :func:`~scikit_pierre.distributions.accessible.distributions_funcs`).

    Returns
    -------
    DataFrame
        Index is ``USER_ID``, columns are unique genre labels, and cell
        values are the distribution probabilities/weights for each user.
    """

    def __map_compute_dist_pandas(user_id) -> DataFrame:
        user_dist_dict = _distribution_component(
            items=_item_in_memory.select_user_items(
                data=users_preference_set[users_preference_set["USER_ID"] == user_id].copy()
            ),
        )
        return DataFrame(
                data=[list(user_dist_dict.values())],
                columns=list(user_dist_dict.keys()),
                index=[user_id]
            )

    _item_in_memory = ItemsInMemory(data=items_df)
    _item_in_memory.item_by_genre()

    _distribution_component = distributions_funcs(distribution=distribution)

    # Cast USER_ID to str for consistent key handling across int/str sources.
    users_preference_set["USER_ID"] = users_preference_set["USER_ID"].astype(str)
    users_ix = users_preference_set["USER_ID"].unique().tolist()

    users_pref_dist_list = list(map(__map_compute_dist_pandas, users_ix))

    return concat(users_pref_dist_list, ignore_index=False)


def computer_users_distribution_dict(
        interactions_df: DataFrame, items_df: DataFrame, distribution: str
) -> dict:
    """
    Compute per-user genre distributions and return the result as a nested dict.

    Parameters
    ----------
    interactions_df : DataFrame
        User interaction data.  Expected columns:
        ``USER_ID``, ``ITEM_ID``, ``TRANSACTION_VALUE``, and optionally
        ``TIMESTAMP``.
    items_df : DataFrame
        Item catalogue with at least ``ITEM_ID`` and ``GENRES`` columns.
    distribution : str
        Acronym identifying the distribution strategy (see
        :func:`~scikit_pierre.distributions.accessible.distributions_funcs`).

    Returns
    -------
    dict
        Mapping of ``user_id -> {genre: probability_value}`` for every
        unique user in *interactions_df*.
    """
    _item_in_memory = ItemsInMemory(data=items_df)
    _item_in_memory.item_by_genre()

    _distribution_component = distributions_funcs(distribution=distribution)

    return_dict = {
        user_id: _distribution_component(
            items=_item_in_memory.select_user_items(
                data=interactions_df[interactions_df["USER_ID"] == user_id].copy()
            ),
        )
        for user_id in interactions_df["USER_ID"].unique().tolist()
    }

    return return_dict


def transform_to_vec(target_dist: dict, realized_dist: dict):
    """
    Align two genre-keyed distribution dicts into parallel numeric vectors.

    The union of genre keys from both dicts is used as the common feature
    space; genres absent from one dict receive a value of 0.0.

    Parameters
    ----------
    target_dist : dict
        Target distribution mapping ``genre -> float``.
    realized_dist : dict
        Realized distribution mapping ``genre -> float``.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(p, q)`` — paired float lists aligned over the union of keys.
        ``p`` corresponds to *target_dist* and ``q`` to *realized_dist*.
    """

    def __map_transform_to_vec(column):
        """
        Return the (target, realized) pair for a single genre column.

        Parameters
        ----------
        column : str
            Genre label to look up in both dicts.

        Returns
        -------
        tuple[float, float]
            ``(p_value, q_value)``; 0.0 when the label is absent.
        """
        if column in target_dist:
            pc = float(target_dist[str(column)])
        else:
            pc = 0.0

        if column in realized_dist:
            qc = float(realized_dist[str(column)])
        else:
            qc = 0.0

        return pc, qc

    columns_list = list(set(list(target_dist.keys()) + list(realized_dist.keys())))
    dist_tuple = list(map(__map_transform_to_vec, columns_list))
    p = [t[0] for t in dist_tuple]
    q = [t[1] for t in dist_tuple]

    return p, q

