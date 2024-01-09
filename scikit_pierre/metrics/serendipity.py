from pandas import DataFrame


def serendipity(users_recommendation_list: DataFrame, users_test_items: DataFrame, users_baseline_items: DataFrame) -> float:
    """
    Serendipity

    :param users_recommendation_list: A Pandas DataFrame, which represents the users recommendation lists.
    :param users_test_items: A Pandas DataFrame, which represents the test items for the experiment.
    :param users_baseline_items:

    :return: A float, which represents the serendipity value.
    """

    def srdp(rec_items: tuple, test_items: tuple, baseline_items: tuple) -> float:
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()
        baselines_items_ids = baseline_items[1]['ITEM_ID'].tolist()

        useful = list(set(rec_items_ids) & set(test_items_ids))

        unexpected_ids = list(set(rec_items_ids) - set(baselines_items_ids))

        sen = list(set(unexpected_ids) & set(useful))

        n_unexpected = 0
        if len(sen) > 0 and len(rec_items_ids) > 0:
            n_unexpected = len(sen) / len(rec_items_ids)
        return n_unexpected

    users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
    users_test_items.sort_values(by=['USER_ID'], inplace=True)
    users_baseline_items.sort_values(by=['USER_ID'], inplace=True)

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(users_test_items['USER_ID'].unique().tolist()):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])
    baseline_set = users_baseline_items.groupby(by=['USER_ID'])

    users_results = list(map(
        srdp,
        rec_set,
        test_set,
        baseline_set
    ))
    return sum(users_results)/len(users_results)
