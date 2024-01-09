from pandas import DataFrame


def surprise(users_recommendation_list: DataFrame, users_preference_items: DataFrame) -> float:
    """
    Serendipity

    :param users_recommendation_list: A Pandas DataFrame, which represents the users recommendation lists.
    :param users_preference_items:

    :return: A float, which represents the surprise value.
    """

    def surp(rec_items: tuple) -> float:
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        return 0

    users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
    users_preference_items.sort_values(by=['USER_ID'], inplace=True)

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(users_preference_items['USER_ID'].unique().tolist()):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    preference_set = users_preference_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])

    users_results = list(map(
        surp,
        rec_set
    ))
    return sum(users_results)/len(users_results)
