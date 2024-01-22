from pandas import DataFrame

from scikit_pierre.distributions.accessible import distributions_funcs
from scikit_pierre.distributions.compute_distribution import computer_users_distribution
from scikit_pierre.distributions.compute_tilde_q import compute_tilde_q
from scikit_pierre.models.item import ItemsInMemory


#########################################################################################
def mean_average_precision(users_recommendation_list: DataFrame, users_test_items: DataFrame) -> float:
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


######################################################################
def mean_reciprocal_rank(users_recommendation_list: DataFrame, users_test_items: DataFrame) -> float:
    def get_rr_from_list(relevance_array: list) -> float:
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                return 1 / (i + 1)
        return 0.0

    def reciprocal_rank(rec_items, test_items) -> float:
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()
        precision = [True if x in test_items_ids else False for x in rec_items_ids]
        return get_rr_from_list(precision)

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


#####################################################################
def mace(
        users_recommendation_lists: DataFrame, items_set_df: DataFrame,
        distribution: str,
        users_preference_set: DataFrame
) -> float:
    """
    Mean Average Calibration Error. Metric to calibrated recommendations systems.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param users_preference_set: TODO: Docstring
    :param users_recommendation_lists: A Pandas DataFrame, which represents the users recommendation lists.
    :param items_set_df: A Dataframe were the lines are the items, the columns are the classes and the cells are probability values.
    :param distribution: A calibration function name.

    :return: A float that's represents the mace value.
    """
    def __intermediate__(user_rec_list_df) -> DataFrame:
        user_rec_list_dict = _item_in_memory.select_user_items(data=user_rec_list_df)
        user_dist_dict = _distribution_component(
            items=user_rec_list_dict,
        )
        return DataFrame([list(user_dist_dict.values())], columns=list(user_dist_dict.keys()))

    def __calibration_error(target_dist: DataFrame, realized_dist: DataFrame):
        columns = target_dist.columns.tolist() + realized_dist.columns.tolist()
        diff_result = []
        for column in columns:
            try:
                t_value = float(target_dist[column].iloc[0])
            except Exception as e:
                t_value = 0.00001

            try:
                r_value = float(realized_dist[column].iloc[0])
            except Exception as e:
                r_value = 0.00001

            diff_result.append(abs(t_value - r_value))
        return sum(diff_result) / len(diff_result)

    def __ace(user_target_distribution: DataFrame, user_rec_list_df: DataFrame):
        user_rec_list_df.sort_values(by=['ORDER'], inplace=True)
        result_ace = [
            __calibration_error(
                target_dist=user_target_distribution.to_frame().reset_index(),
                realized_dist=__intermediate__(
                    user_rec_list_df=user_rec_list_df.head(k)
                )
            ) for k in user_rec_list_df['ORDER'].tolist()
        ]
        return sum(result_ace) / len(result_ace)

    _item_in_memory = ItemsInMemory(data=items_set_df)
    _item_in_memory.item_by_genre()
    _distribution_component = distributions_funcs(distribution=distribution)

    users_target_dist = computer_users_distribution(
        users_preference_set=users_preference_set, items_df=items_set_df, distribution=distribution
    )
    users_target_dist.sort_index(inplace=True)
    users_target_dist.fillna(0, inplace=True)
    users_recommendation_lists.sort_values(by=['USER_ID'], inplace=True)

    set_1 = set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()])
    set_2 = set([str(ix) for ix in users_target_dist.index])

    if set_1 != set_2:
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    results = list(map(
        lambda utarget_dist, urec_list: __ace(
            user_target_distribution=utarget_dist[1], user_rec_list_df=urec_list[1]
        ),
        users_target_dist.iterrows(),
        users_recommendation_lists.groupby(by=['USER_ID'])
    ))
    return sum(results) / len(results)


#######################################################

def mrmc(users_target_dist, users_recommendation_lists, items_classes_set, dist_func, fairness_func):
    """
    Mean Rank MisCalibration. Metric to calibrated recommendations systems.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

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

    def __rank_miscalibration(user_id, user_target_distribution, user_rec_list):
        user_rec_list.sort_values(by=['ORDER'])
        result = [
            __miscalibration(
                target_dist=user_target_distribution,
                realized_dist=dist_func(
                    user_id=user_id,
                    user_pref_set=user_rec_list.head(k),
                    item_classes_set=items_classes_set
                )
            ) for k in user_rec_list['ORDER'].tolist()
        ]
        return sum(result) / len(result)

    users_recommendation_lists.sort_values(by=['USER_ID'], inplace=True)
    users_target_dist.sort_index(inplace=True)

    if set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()]) != set([str(ix) for ix in users_target_dist.index]):
        raise Exception('Unknown users in recommendation or test set. Please make sure the users are the same.')

    results = list(map(
        lambda utarget_dist, urec_list: __rank_miscalibration(
            user_target_distribution=utarget_dist[1], user_rec_list=urec_list[1], user_id=urec_list[0]
        ),
        users_target_dist.iterrows(),
        users_recommendation_lists.groupby(by=['USER_ID'])
    ))
    return sum(results) / len(results)


####################################################
def gap(users_data: DataFrame) -> float:
    uuids = users_data['USER_ID'].unique()

    numerator = 0
    denominator = len(uuids)

    for uid in uuids:
        user_pref = users_data[users_data['USER_ID'] == uid]

        numerator += user_pref['popularity'].mean()

    return numerator / denominator


def popularity_lift(users_model: DataFrame, users_recommendations: DataFrame) -> float:
    """
    Positive values for PL indicate amplification of popularity bias by the algorithm.
    A negative value for PL happens when, on average, the recommendations are less concentrated on popular items than the users’ profile.
    Moreover, the PL value of 0 means there is no popularity bias amplification.
    :param users_model:
    :param users_recommendations:
    :return:
    """
    q = gap(users_recommendations)
    p = gap(users_model)
    return (q - p) / p


################################################################
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


###########################################################

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


#######################################################
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
