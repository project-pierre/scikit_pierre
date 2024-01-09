from pandas import DataFrame


def unexpectedness(users_recommendation_list: DataFrame, users_test_items: DataFrame) -> float:
    """
    Serendipity

    :param users_recommendation_list: A Pandas DataFrame, which represents the users recommendation lists.
    :param users_test_items: A Pandas DataFrame, which represents the test items for the experiment.

    :return: A float, which represents the map value.
    """

    def unex(rec_items: tuple, test_items: tuple) -> float:
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()

        unexpected_ids = list(set(rec_items_ids) - set(test_items_ids))
        n_unexpected = len(unexpected_ids) / len(rec_items_ids)
        return n_unexpected

    users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
    users_test_items.sort_values(by=['USER_ID'], inplace=True)

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(users_test_items['USER_ID'].unique().tolist()):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])

    users_results = list(map(
        unex,
        rec_set,
        test_set
    ))
    return sum(users_results)/len(users_results)
