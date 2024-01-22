from pandas import DataFrame

from scikit_pierre.distributions.accessible import distributions_funcs_pandas


def mace(users_target_dist: DataFrame, users_recommendation_lists: DataFrame, items_classes_set: DataFrame, distribution: str) -> float:
    """
    Mean Average Calibration Error. Metric to calibrated recommendations systems.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param users_target_dist: A DataFrame were the lines are the users, the columns are the classes and the cells are the distribution value.
    :param users_recommendation_lists: A Pandas DataFrame, which represents the users recommendation lists.
    :param items_classes_set: A Dataframe were the lines are the items, the columns are the classes and the cells are probability values.
    :param distribution: A calibration function name.

    :return: A float that's represents the mace value.
    """
    def __calibration_error(target_dist: DataFrame, realized_dist: DataFrame):
        diff_result = [abs(float(target_dist[column] - float(realized_dist[column])))
                       for column in realized_dist]
        return sum(diff_result) / len(diff_result)

    def __ace(user_id: str, user_target_distribution: DataFrame, user_rec_list: DataFrame):
        user_rec_list.sort_values(by=['ORDER'], inplace=True)
        result_ace = [
            __calibration_error(
                target_dist=user_target_distribution,
                realized_dist=dist_func(
                    user_id=user_id,
                    user_pref_set=user_rec_list.head(k),
                    item_classes_set=items_classes_set)
            ) for k in user_rec_list['ORDER'].tolist()
        ]
        return sum(result_ace) / len(result_ace)

    dist_func = distributions_funcs_pandas(distribution)
    users_recommendation_lists.sort_values(by=['USER_ID'], inplace=True)
    users_target_dist.sort_index(inplace=True)

    if set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()]) != set([str(ix) for ix in users_target_dist.index]):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    results = list(map(
        lambda utarget_dist, urec_list: __ace(
            user_id=urec_list[0], user_target_distribution=utarget_dist[1], user_rec_list=urec_list[1]
        ),
        users_target_dist.iterrows(),
        users_recommendation_lists.groupby(by=['USER_ID'])
    ))
    return sum(results) / len(results)
