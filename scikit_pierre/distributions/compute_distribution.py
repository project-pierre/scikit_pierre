from pandas import DataFrame, concat

from scikit_pierre.distributions.accessible import distributions_funcs
from scikit_pierre.models.item import ItemsInMemory


def computer_users_distribution(users_preference_set: DataFrame, items_df: DataFrame, distribution: str) -> DataFrame:
    # Get the items classes
    _item_in_memory = ItemsInMemory(data=items_df)
    _item_in_memory.item_by_genre()

    # Set the used distribution
    _distribution_component = distributions_funcs(distribution=distribution)

    # Group the preferences by user
    grouped_users_preference_set = users_preference_set.groupby(by=["USER_ID"])

    # Compute the distribution to all users
    users_pref_dist_list = []
    for user_id, user_pref_set in grouped_users_preference_set:
        user_dist_dict = _distribution_component(
            items=_item_in_memory.select_user_items(data=user_pref_set),
        )
        users_pref_dist_list.append(
            DataFrame([list(user_dist_dict.values())], columns=list(user_dist_dict.keys()), index=user_id)
        )

    return concat(users_pref_dist_list)
