from pandas import DataFrame


def mean_average_precision_map(users_recommendation_list: DataFrame, users_test_items: DataFrame) -> float:
    """
    Mean Average Precision. A metric to get the precision along the recommendation list.

    :param users_recommendation_list: A Pandas DataFrame, which represents the users recommendation lists.
    :param users_test_items: A Pandas DataFrame, which represents the test items for the experiment.

    :return: A float, which represents the map value.
    """

    def get_ap_from_list(relevance_array: list) -> float:
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        hit_list = []
        relevant = 0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                relevant += 1
            hit_list.append(relevant / (i + 1))
        ap = sum(hit_list)
        if ap > 0.0:
            return ap / relevance_list_size
        else:
            return 0.0

    def average_precision(rec_items: tuple, test_items: tuple) -> float:
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()
        precision = [True if x in test_items_ids else False for x in rec_items_ids]
        return get_ap_from_list(precision)

    users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
    users_test_items.sort_values(by=['USER_ID'], inplace=True)

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(users_test_items['USER_ID'].unique().tolist()):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])

    users_results = list(map(
        average_precision,
        rec_set,
        test_set
    ))
    return sum(users_results)/len(users_results)
