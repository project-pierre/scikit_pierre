from scikit_pierre.pierre.distributions.compute_tilde_q import compute_tilde_q


def mrmc(users_target_dist, users_recommendation_lists, items_classes_set, dist_func, fairness_func):
    """
    Mean Rank MisCalibration. Metric to calibrated recommendations systems.

    Implementation based on:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param users_target_dist: A DataFrame were the lines are the users, the columns are the classes and the cells are the distribution value.
    :param users_recommendation_lists: A Pandas DataFrame, which represents the users recommendation lists.
    :param items_classes_set: A Dataframe were the lines are the items, the columns are the classes and the cells are probability values.
    :param fairness_func: A fairness function.
    :param dist_func: ...

    :return: A float that's represents the mace value.
    """
    def __miscalibration(target_dist, realized_dist):
        p = list(target_dist)
        q = list(realized_dist.values[0])
        tild = compute_tilde_q(p=p, q=q)
        numerator = fairness_func(p=p, q=tild)
        denominator = fairness_func(p=p, q=[0.00001 for _ in range(len(p))])
        try:
            return abs(numerator / denominator)
        except Exception as e:
            if numerator is None or numerator == [] or numerator == 0.0:
                numerator = 0.00001
        return abs(numerator / denominator)

    def __rank_miscalibration(user_target_distribution, user_rec_list):
        user_rec_list.sort_values(by=['ORDER'])
        result = [
            __miscalibration(
                target_dist=user_target_distribution,
                realized_dist=dist_func(
                    user_pref_set=user_rec_list.head(k),
                    item_classes_set=items_classes_set
                )
            ) for k in user_rec_list['ORDER'].tolist()
        ]
        return sum(result) / len(result)

    users_recommendation_lists.sort_values(by=['USER_ID'], inplace=True)
    users_target_dist.sort_index(inplace=True)

    if set(users_recommendation_lists['USER_ID'].unique().tolist()) != set(users_target_dist.index):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    results = list(map(
        lambda utarget_dist, urec_list: __rank_miscalibration(utarget_dist[1], urec_list[1]),
        users_target_dist.iterrows(),
        users_recommendation_lists.groupby(by=['USER_ID'])
    ))
    return sum(results) / len(results)
