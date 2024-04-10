"""
File to transform Dataframe in Item class and Item class in Dataframe.
"""
from pandas import DataFrame, concat

from .accessible import distributions_funcs
from ..models.item import ItemsInMemory


def computer_users_distribution_pandas(
        users_preference_set: DataFrame, items_df: DataFrame, distribution: str
) -> DataFrame:
    """

    :param users_preference_set: A Pandas DataFrame with four columns
                                [USER_ID, ITEM_ID, TRANSACTION_VALUE, TIMESTAMP].
    :param items_df: A Pandas DataFrame of items with two columns
                    [ITEM_ID, GENRES].
    :param distribution: The string name of the used distribution.
    :return: A Pandas DataFrame with the USER_ID as index, GENRES as columns,
            and the distribution value as cells.
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
                index=[str(user_id)]
            )

    # Get the items classes
    _item_in_memory = ItemsInMemory(data=items_df)
    _item_in_memory.item_by_genre()

    # Set the used distribution
    _distribution_component = distributions_funcs(distribution=distribution)

    # Group the preferences by user
    users_preference_set["USER_ID"] = users_preference_set["USER_ID"].astype(str)
    users_ix = users_preference_set["USER_ID"].unique().tolist()

    # Compute the distribution to all users
    users_pref_dist_list = list(map(__map_compute_dist_pandas, users_ix))

    # users_pref_dist_list = []
    # for user_id in users_ix:
    #     user_dist_dict = _distribution_component(
    #         items=_item_in_memory.select_user_items(
    #             data=users_preference_set[users_preference_set["USER_ID"] == user_id].copy()
    #         ),
    #     )
    #     users_pref_dist_list.append(
    #         DataFrame(
    #             data=[list(user_dist_dict.values())],
    #             columns=list(user_dist_dict.keys()),
    #             index=[str(user_id)]
    #         )
    #     )

    return concat(users_pref_dist_list, ignore_index=False)


def computer_users_distribution_dict(
        interactions_df: DataFrame, items_df: DataFrame, distribution: str
) -> dict:
    """

    :param interactions_df: A Pandas DataFrame with four columns
                                [USER_ID, ITEM_ID, TRANSACTION_VALUE, TIMESTAMP].
    :param items_df: A Pandas DataFrame of items with two columns
                    [ITEM_ID, GENRES].
    :param distribution: The string name of the used distribution.
    :return: Dict
    """
    # Get the items classes
    _item_in_memory = ItemsInMemory(data=items_df)
    _item_in_memory.item_by_genre()

    # Set the used distribution
    _distribution_component = distributions_funcs(distribution=distribution)

    # Compute the distribution to all users
    return_dict = {
        str(user_id): _distribution_component(
            items=_item_in_memory.select_user_items(
                data=interactions_df[interactions_df["USER_ID"] == user_id].copy()
            ),
        )
        for user_id in interactions_df["USER_ID"].unique().tolist()
    }

    return return_dict


def transform_to_vec(target_dist: dict, realized_dist: dict):
    """

    :param target_dist:
    :param realized_dist:
    :return:
    """

    def __map_transform_to_vec(column):
        """

        :param column:
        :return:
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

