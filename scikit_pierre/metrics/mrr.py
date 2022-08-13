def get_rr_from_list(relevance_array):
    relevance_list_size = len(relevance_array)
    if relevance_list_size == 0:
        return 0.0
    for i in range(relevance_list_size):
        if relevance_array[i]:
            return 1 / (i + 1)
    return 0.0


def reciprocal_rank(rec_items, test_items):
    rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
    test_items_ids = test_items[1]['ITEM_ID'].tolist()
    precision = [True if x in test_items_ids else False for x in rec_items_ids]
    return get_rr_from_list(precision)


def mean_reciprocal_rank(users_recommendation_list, users_test_items):
    users_ids = users_recommendation_list['USER_ID'].unique().tolist()
    users_results = [reciprocal_rank(users_recommendation_list[users_recommendation_list['USER_ID'] == user_id],
                                     users_test_items[users_test_items['USER_ID'] == user_id])
                     for user_id in users_ids]
    return sum(users_results)/len(users_results)


def mean_reciprocal_rank_map(users_recommendation_list, users_test_items):
    users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
    users_test_items.sort_values(by=['USER_ID'], inplace=True)

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(users_test_items['USER_ID'].unique().tolist()):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])

    users_results = list(map(
        reciprocal_rank,
        rec_set,
        test_set
    ))
    return sum(users_results)/len(users_results)
