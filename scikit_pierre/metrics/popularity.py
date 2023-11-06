from pandas import DataFrame


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
