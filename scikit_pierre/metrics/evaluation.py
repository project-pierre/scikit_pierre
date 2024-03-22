"""
This file contains all evaluation metrics.
"""
from numpy import mean
from pandas import DataFrame

from ..distributions.accessible import distributions_funcs
from ..distributions.compute_distribution import computer_users_distribution_dict
from ..distributions.compute_tilde_q import compute_tilde_q
from ..measures.accessible import calibration_measures_funcs
from ..models.item import ItemsInMemory


class BaseMetric:
    """

    """

    def __init__(self, users_profile_df: DataFrame, users_rec_list_df: DataFrame):
        """

        """
        self.profiles_df = users_profile_df
        self.rec_df = users_rec_list_df

        self.grouped_profiles_df = None
        self.grouped_rec_df = None

    def checking_users(self):
        """

        :return:
        """
        set_1 = set({str(ix) for ix in self.profiles_df['USER_ID'].unique().tolist()})
        set_2 = set({str(ix) for ix in self.rec_df['USER_ID'].unique().tolist()})

        if set_1 != set_2:
            raise IndexError(
                'Unknown users in recommendation or test set. '
                'Please make sure the users are the same.'
            )

    @staticmethod
    def get_bool_list(rec_items: tuple, test_items: tuple) -> list:
        """

        :param rec_items:
        :param test_items:
        :return:
        """
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()
        return [x in test_items_ids for x in rec_items_ids]

    def ordering(self) -> None:
        """

        :return:
        """
        self.profiles_df.sort_values(by=['USER_ID'], inplace=True)
        self.rec_df.sort_values(by=['USER_ID'], inplace=True)

    def grouping(self) -> None:
        """

        :return:
        """
        self.grouped_profiles_df = self.profiles_df.groupby(by=['USER_ID'])
        self.grouped_rec_df = self.rec_df.groupby(by=['USER_ID'])

    def ordering_and_grouping(self) -> None:
        """

        :return:
        """
        self.ordering()
        self.grouping()


# ################################################################################################ #
# ######################################## Accuracy Metrics ###################################### #
# ################################################################################################ #
class MeanAveragePrecision(BaseMetric):
    """
    Mean Average Precision. A metric to get the precision along the recommendation list.

    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_set_df: DataFrame):
        """

        :param users_rec_list_df: A Pandas DataFrame,
        which represents the users recommendation lists.

        :param users_test_set_df: A Pandas DataFrame,
        which represents the test items for the experiment.
        """
        super().__init__(users_profile_df=users_test_set_df, users_rec_list_df=users_rec_list_df)

    @staticmethod
    def get_list_precision(relevance_array: list) -> float:
        """

        :param relevance_array:
        :return:
        """
        if len(relevance_array) == 0:
            return 0.0
        hit_list = []
        relevant = 0
        for i, value in enumerate(relevance_array):
            if value:
                relevant += 1
            hit_list.append(relevant / (i + 1))
        return mean(hit_list)

    def average_precision(self, rec_items: tuple, test_items: tuple) -> float:
        """

        :param rec_items:
        :param test_items:
        :return:
        """
        return self.get_list_precision(
            self.get_bool_list(rec_items=rec_items, test_items=test_items)
        )

    def compute(self) -> float:
        """

        :return:
        """
        self.checking_users()
        self.ordering_and_grouping()

        users_results = list(map(
            self.average_precision,
            self.grouped_rec_df,
            self.grouped_profiles_df
        ))
        return mean(users_results)


class MeanReciprocalRank(BaseMetric):
    """
    Mean Reciprocal Rank.

    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_set_df: DataFrame):
        """

        :param users_rec_list_df: A Pandas DataFrame,
        which represents the users recommendation lists.

        :param users_test_set_df: A Pandas DataFrame,
        which represents the test items for the experiment.
        """
        super().__init__(users_profile_df=users_test_set_df, users_rec_list_df=users_rec_list_df)

    @staticmethod
    def get_list_reciprocal(relevance_array: list) -> float:
        """

        :param relevance_array:
        :return:
        """
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i, value in enumerate(relevance_array):
            if value:
                return 1 / (i + 1)
        return 0.0

    def average_reciprocal(self, rec_items: tuple, test_items: tuple) -> float:
        """

        :param rec_items:
        :param test_items:
        :return:
        """
        return self.get_list_reciprocal(
            self.get_bool_list(rec_items=rec_items, test_items=test_items)
        )

    def compute(self) -> float:
        """

        :return:
        """
        self.checking_users()
        self.ordering_and_grouping()

        users_results = list(map(
            self.average_reciprocal,
            self.grouped_rec_df,
            self.grouped_profiles_df
        ))
        return mean(users_results)


# ################################################################################################ #
# ###################################### Calibration Metrics ##################################### #
# ################################################################################################ #
class BaseCalibrationMetric(BaseMetric):
    """
    Base calibration metric class.
    """
    def __init__(
            self,
            users_profile_df: DataFrame, users_rec_list_df: DataFrame, items_set_df: DataFrame,
            distribution_name: str = "CWS", distance_func_name: str = "KL"
    ):
        """

        :param users_profile_df:
        :param users_rec_list_df:
        :param items_set_df:
        :param distribution_name:
        :param distance_func_name:
        """
        super().__init__(users_profile_df=users_profile_df, users_rec_list_df=users_rec_list_df)
        self.target_dist = None
        self.realized_dist = None

        self.items_df = items_set_df
        self._item_in_memory = None

        self.dist_func = distributions_funcs(distribution=distribution_name)
        self.dist_name = distribution_name

        self.calib_measure_func = calibration_measures_funcs(measure=distance_func_name)
        self.calib_measure_name = distance_func_name

    def item_preparation(self) -> None:
        """

        :return:
        """
        self._item_in_memory = ItemsInMemory(data=self.items_df)
        self._item_in_memory.item_by_genre()

    @staticmethod
    def transform_to_vec(target_dist, realized_dist):
        """

        :param target_dist:
        :param realized_dist:
        :return:
        """
        p = []
        q = []
        columns_list = list(set(list(target_dist.keys()) + list(realized_dist.keys())))
        for column in columns_list:
            if column in target_dist:
                p.append(float(target_dist[str(column)]))
            else:
                p.append(0.00001)

            if column in realized_dist:
                q.append(float(realized_dist[str(column)]))
            else:
                q.append(0.00001)
        return p, q

    def compute_distribution(self, set_df: DataFrame) -> dict:
        """

        :param set_df:
        :return:
        """
        dist_dict = computer_users_distribution_dict(
            interactions_df=set_df, items_df=self.items_df,
            distribution=self.dist_name
        )
        return dist_dict

    def compute(self):
        """

        :return:
        """
        self.checking_users()
        self.target_dist = self.compute_distribution(self.profiles_df)


class MeanAbsoluteCalibrationError(BaseCalibrationMetric):
    """
    Mean Absolute Calibration Error. Metric to calibrated recommendations systems.

    Implementation based on:
    - Exploiting personalized calibration and metrics for fairness recommendation -
    Silva et al. (2021) - https://doi.org/10.1016/j.eswa.2021.115112

    """

    def compute_ace(self, target_dist: dict, realized_dist: dict) -> float:
        """

        :param target_dist:
        :param realized_dist:
        :return:
        """
        p, q = self.transform_to_vec(target_dist, realized_dist)
        diff_result = [abs(t_value - r_value) for t_value, r_value in zip(p, q)]
        return mean(diff_result)

    def based_on_position(self, rec_pos_df: DataFrame, user_indexes: list) -> float:
        """

        :param user_indexes:
        :param rec_pos_df:
        :return:
        """
        self.realized_dist = self.compute_distribution(rec_pos_df)
        results = [
            self.compute_ace(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)],
            ) for ix in user_indexes
        ]
        return mean(results)

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        list_size = self.rec_df["ORDER"].max()

        user_indexes = list(self.target_dist.keys())
        results = [
            self.based_on_position(
                rec_pos_df=self.rec_df[self.rec_df["ORDER"] <= i].copy(),
                user_indexes=user_indexes
            ) for i in range(1, list_size + 1)
        ]
        return mean(results)


class Miscalibration(BaseCalibrationMetric):
    """
    Miscalibration. Metric to calibrated recommendations systems.

    Implementation based on:
    - Calibrated Recommendations - Steck (2018) - https://doi.org/10.1145/3240323.3240372

    """

    def compute_miscalibration(self, target_dist, realized_dist) -> float:
        """

        :param target_dist:
        :param realized_dist:
        :return:
        """
        p, q = self.transform_to_vec(target_dist, realized_dist)
        return self.calib_measure_func(
            p=p,
            q=compute_tilde_q(p=p, q=q)
        )

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        self.realized_dist = self.compute_distribution(self.rec_df)

        user_indexes = list(self.target_dist.keys())

        results = [
            self.compute_miscalibration(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)]
            )
            for ix in user_indexes
        ]

        return mean(results)


class MeanAverageMiscalibration(Miscalibration):
    """
    Mean Average Miscalibration. Metric to calibrated recommendations systems.

    Implementation based on:
    -

    """

    def based_on_position(self, rec_pos_df: DataFrame, user_indexes: list) -> float:
        """

        :param user_indexes:
        :param rec_pos_df:
        :return:
        """
        self.realized_dist = self.compute_distribution(rec_pos_df)
        results = [
            self.compute_miscalibration(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)],
            ) for ix in user_indexes
        ]
        return mean(results)

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        list_size = self.rec_df["ORDER"].max()

        user_indexes = list(self.target_dist.keys())
        results = [
            self.based_on_position(
                rec_pos_df=self.rec_df[self.rec_df["ORDER"] <= i].copy(),
                user_indexes=user_indexes
            ) for i in range(1, list_size + 1)
        ]
        return mean(results)


#######################################################

def mrmc(users_target_dist, users_rec_list_df, items_classes_set, dist_func,
         fairness_func):
    """
    Mean Rank MisCalibration. Metric to calibrated recommendations systems.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    :param users_target_dist:
        A DataFrame were the lines are the users,
        the columns are the classes and the cells are the distribution value.
    :param users_rec_list_df:
        A Pandas DataFrame, which represents the users recommendation lists.
    :param items_classes_set:
        A Dataframe were the lines are the items,
        the columns are the classes and the cells are probability values.
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
        except (ArithmeticError, ZeroDivisionError, KeyError):
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

    users_rec_list_df.sort_values(by=['USER_ID'], inplace=True)
    users_target_dist.sort_index(inplace=True)

    if set({str(ix) for ix in users_rec_list_df['USER_ID'].unique().tolist()}) != set(
            {str(ix) for ix in users_target_dist.index}):
        raise IndexError(
            'Unknown users in recommendation or test set. Please make sure the users are the same.')

    results = list(map(
        lambda utarget_dist, urec_list: __rank_miscalibration(
            user_target_distribution=utarget_dist[1], user_rec_list=urec_list[1],
            user_id=urec_list[0]
        ),
        users_target_dist.iterrows(),
        users_rec_list_df.groupby(by=['USER_ID'])
    ))
    return sum(results) / len(results)


# ################################################################################################ #
# ###################################### Popularity Metrics ###################################### #
# ################################################################################################ #
def gap(users_data: DataFrame) -> float:
    """
    GAP function
    :param users_data:
    :return:
    """
    uuids = users_data['USER_ID'].unique()

    numerator = 0
    denominator = len(uuids)

    for uid in uuids:
        user_pref = users_data[users_data['USER_ID'] == uid]

        numerator += float(user_pref['popularity'].mean())

    return numerator / denominator


def popularity_lift(users_model: DataFrame, users_recommendations: DataFrame) -> float:
    """
    Positive values for PL indicate amplification of popularity bias by the algorithm.
    A negative value for PL happens when, on average,
    the recommendations are less concentrated on popular items than the users’ profile.
    Moreover, the PL value of 0 means there is no popularity bias amplification.
    :param users_model:
    :param users_recommendations:
    :return:
    """
    q = gap(users_recommendations)
    p = gap(users_model)
    return (q - p) / p


################################################################
def serendipity(users_recommendation_list: DataFrame, users_test_items: DataFrame,
                users_baseline_items: DataFrame) -> float:
    """
    Serendipity

    :param users_recommendation_list:
        A Pandas DataFrame, which represents the users recommendation lists.
    :param users_test_items:
        A Pandas DataFrame, which represents the test items for the experiment.
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

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(
            users_test_items['USER_ID'].unique().tolist()):
        raise IndexError(
            'Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])
    baseline_set = users_baseline_items.groupby(by=['USER_ID'])

    users_results = list(map(
        srdp,
        rec_set,
        test_set,
        baseline_set
    ))
    return sum(users_results) / len(users_results)


###########################################################

# def surprise(users_recommendation_list: DataFrame, users_preference_items: DataFrame) -> float:
#     """
#     Serendipity
#
#     :param users_recommendation_list:
#         A Pandas DataFrame, which represents the users recommendation lists.
#     :param users_preference_items:
#
#     :return: A float, which represents the surprise value.
#     """
#
#     def surp(rec_items: tuple) -> float:
#         rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
#         return 0
#
#     users_recommendation_list.sort_values(by=['USER_ID'], inplace=True)
#     users_preference_items.sort_values(by=['USER_ID'], inplace=True)
#
#     if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(
#             users_preference_items['USER_ID'].unique().tolist()):
#         raise IndexError(
#             'Unknown users in recommendation or test set.
#             Please make sure the users are the same.')
#
#     # preference_set = users_preference_items.groupby(by=['USER_ID'])
#     rec_set = users_recommendation_list.groupby(by=['USER_ID'])
#
#     users_results = list(map(
#         surp,
#         rec_set
#     ))
#     return sum(users_results) / len(users_results)


#######################################################
def unexpectedness(users_recommendation_list: DataFrame, users_test_items: DataFrame) -> float:
    """
    Serendipity

    :param users_recommendation_list: A Pandas DataFrame,
        which represents the users recommendation lists.
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

    if set(users_recommendation_list['USER_ID'].unique().tolist()) != set(
            users_test_items['USER_ID'].unique().tolist()):
        raise IndexError(
            'Unknown users in recommendation or test set. Please make sure the users are the same.')

    test_set = users_test_items.groupby(by=['USER_ID'])
    rec_set = users_recommendation_list.groupby(by=['USER_ID'])

    users_results = list(map(
        unex,
        rec_set,
        test_set
    ))
    return sum(users_results) / len(users_results)
