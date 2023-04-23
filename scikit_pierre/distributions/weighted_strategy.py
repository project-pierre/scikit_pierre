from math import log

from pandas import DataFrame


def weighted_strategy_base(item_classes_set: DataFrame, user_pref_set: DataFrame) -> dict:
    """
    The Weighted Strategy - (WS).

    :param item_classes_set: A Dataframe were the lines are the items, the columns are the genres and the cells are probability values.
    :param user_pref_set: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE]

    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute():
        for row in user_pref_set.itertuples():
            for column_name in item_classes_set.columns:
                line = item_classes_set.loc[row.ITEM_ID]
                genre_value = line[column_name]
                if genre_value == 0.0:
                    continue
                numerator[column_name] = numerator.get(column_name, 0.0) + row.TRANSACTION_VALUE * genre_value
                denominator[column_name] = denominator.get(column_name, 0.0) + row.TRANSACTION_VALUE

    def genre(g):
        if (g in denominator.keys() and denominator[g] > 0.0) and (g in numerator.keys() and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        else:
            return 0.0

    compute()
    distribution = {g: genre(g) for g in item_classes_set.columns}
    return distribution


def weighted_strategy(user_pref_set: DataFrame, item_classes_set: DataFrame) -> DataFrame:
    """
    The Weighted Strategy - (WS). The reference for this implementation are from:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045

    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    :param item_classes_set: A Dataframe were the lines are the items, the columns are the genres and the cells are probability values.
    :param user_pref_set: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE]

    :return: A Dataframe with one line. The columns are the genres and the index is the user id. The cells are probability values.
    """
    distribution_dict = weighted_strategy_base(item_classes_set, user_pref_set)
    user_id = user_pref_set.iloc[0]['USER_ID']
    distribution = DataFrame.from_records(distribution_dict, index=[user_id])
    return distribution.fillna(0.0)


def weighted_probability_strategy(item_classes_set: DataFrame, user_pref_set: DataFrame) -> DataFrame:
    """
    The Weighted Probability Strategy - (WPS). The reference for this implementation are from:

    - Silva and Durão (2022). https://arxiv.org/abs/2204.03706

    :param item_classes_set: A Dataframe were the lines are the items, the columns are the genres and the cells are probability values.
    :param user_pref_set: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE]

    :return: A Dataframe with one line. The columns are the genres and the index is the user id. The cells are probability values.
    """
    distribution_dict = weighted_strategy_base(item_classes_set, user_pref_set)
    total = sum([value for g, value in distribution_dict.items()])
    user_id = user_pref_set.iloc[0]['USER_ID']
    distribution = DataFrame.from_records({g: value / total for g, value in distribution_dict.items()}, index=[user_id])
    return distribution.fillna(0.0)


def class_ranked_strategy(item_classes_set: DataFrame, user_pref_set: DataFrame) -> DataFrame:
    """
    The Class Ranked Strategy - (CRS). The reference for this implementation are from:

    - Sacharidis, Mouratidis, Kleftogiannis (2019) - https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7946&context=sis_research

    :param item_classes_set: A Dataframe were the lines are the items, the columns are the genres and the cells are probability values.
    :param user_pref_set: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE]

    :return: A Dataframe with one line. The columns are the genres and the index is the user id. The cells are probability values.
    """
    # TODO: REvisar formulação
    filtered = item_classes_set.filter(items = user_pref_set['ITEM_ID'].tolist(), axis=0)

    def constant(column_values):
        return sum(value * (1 / log(ix + 1)) for ix, value in enumerate(column_values, start=1))

    const_value = sum([constant(filtered[column_name].tolist()) for column_name in filtered.columns])

    user_id = user_pref_set.iloc[0]['USER_ID']
    distribution = DataFrame.from_records({column_name: const_value * constant(filtered[column_name].tolist()) for column_name in filtered.columns}, index=[user_id])
    return distribution.fillna(0.0)
