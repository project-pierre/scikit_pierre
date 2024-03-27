"""
This file contains all evaluation metrics.
"""
import itertools

from numpy import mean
from pandas import DataFrame

from ..distributions.accessible import distributions_funcs
from ..distributions.compute_distribution import computer_users_distribution_dict
from ..distributions.compute_tilde_q import compute_tilde_q
from ..measures.accessible import calibration_measures_funcs
from ..models.item import ItemsInMemory


class BaseMetric:
    """
    This is the base class metric to be inherent by all other class metrics.

    - df_1: It is a Pandas Dataframe that can represents: User profile or test items set.

    - df_2: It is a Pandas Dataframe that can represents: User recommendation list.

    - df_3: It is a Pandas Dataframe that can represents: User Candidate items or some baseline.

    The specific meaning depends on the subclass which inherent this super class.
    """

    def __init__(
            self,
            df_1: DataFrame, df_2: DataFrame, df_3: DataFrame = None
    ):
        """

        :param df_1: It is a Pandas Dataframe that can represents: User profile or test items set.

        :param df_2: It is a Pandas Dataframe that can represents: User recommendation list.

        :param df_3: It is a Pandas Dataframe that can represents: Candidate or baseline items.
        """
        self.df_1 = df_1
        self.df_2 = df_2
        self.df_3 = df_3

        self.grouped_df_1 = None
        self.grouped_df_2 = None
        self.grouped_df_3 = None

    def checking_users(self) -> None:
        """
        This method checks if the users ids matches. If it does not match an error is raised.
        """
        set_1 = set({str(ix) for ix in self.df_1['USER_ID'].unique().tolist()})
        set_2 = set({str(ix) for ix in self.df_2['USER_ID'].unique().tolist()})

        if set_1 != set_2:
            raise IndexError(
                'Unknown users in recommendation or test set. '
                'Please make sure the users are the same.'
            )

    @staticmethod
    def get_bool_list(rec_items: tuple, test_items: tuple) -> list:
        """
        This method verify which items are in common in the two tuples.

        :param rec_items: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param test_items: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A list with True or False.
        """
        rec_items_ids = rec_items[1]['ITEM_ID'].tolist()
        test_items_ids = test_items[1]['ITEM_ID'].tolist()
        return [x in test_items_ids for x in rec_items_ids]

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        """
        if self.df_1 is not None:
            self.df_1.sort_values(by=['USER_ID'], inplace=True)

        if self.df_2 is not None:
            self.df_2.sort_values(by=['USER_ID'], inplace=True)

        if self.df_3 is not None:
            self.df_3.sort_values(by=['USER_ID'], inplace=True)

    def grouping(self) -> None:
        """
        This method is for grouping the users lines.
        """
        if self.df_1 is not None:
            self.grouped_df_1 = self.df_1.groupby(by=['USER_ID'])

        if self.df_2 is not None:
            self.grouped_df_2 = self.df_2.groupby(by=['USER_ID'])

        if self.df_3 is not None:
            self.grouped_df_3 = self.df_3.groupby(by=['USER_ID'])

    def ordering_and_grouping(self) -> None:
        """
        This method is to order and group the Dataframe based on the user id.
        It is a guarantee that the users are in the same positions in the interaction.
        """
        self.ordering()
        self.grouping()

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method is a base to be overridden by the subclass.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float with the single computation result.
        """
        pass

    def compute(self) -> float:
        """
        This method is the generic one to start the metric computation.

        :return: A float which comprises the metric value.
        """
        self.checking_users()
        self.ordering_and_grouping()

        users_results = list(map(
            self.single_process,
            self.grouped_df_2,
            self.grouped_df_1
        ))
        return mean(users_results)


# ################################################################################################ #
# ######################################## Accuracy Metrics ###################################### #
# ################################################################################################ #
class MeanAveragePrecision(BaseMetric):
    """
    Mean Average Precision (MAP).
    A metric to get the average precision among all users' recommendation list.

    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_set_df: DataFrame):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_test_set_df: A Pandas DataFrame,
            which represents the test items set for the experiment.
        """
        super().__init__(df_1=users_test_set_df, df_2=users_rec_list_df)

    @staticmethod
    def get_list_precision(relevance_array: list) -> float:
        """
        This method is to compute the precision value of one list.

        :param relevance_array: A list with True or False in the positions.

        :return: A float which comprises the metric value from the relevance array.
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

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (MAP) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (MAP) value for one user.
        """
        return self.get_list_precision(
            relevance_array=self.get_bool_list(
                rec_items=tuple_from_df_2, test_items=tuple_from_df_1
            )
        )


class MeanReciprocalRank(BaseMetric):
    """
    Mean Reciprocal Rank (MRR).

    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_set_df: DataFrame):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_test_set_df: A Pandas DataFrame,
            which represents the test items set for the experiment.
        """
        super().__init__(df_1=users_test_set_df, df_2=users_rec_list_df)

    @staticmethod
    def get_list_reciprocal(relevance_array: list) -> float:
        """
        This method is to compute the reciprocal value of one list.

        :param relevance_array: A list with True or False in the positions.

        :return: A float which comprises the metric value from the relevance array.
        """
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i, value in enumerate(relevance_array):
            if value:
                return 1 / (i + 1)
        return 0.0

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (MRR) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (MRR) value for one user.
        """
        return self.get_list_reciprocal(
            relevance_array=self.get_bool_list(
                rec_items=tuple_from_df_2, test_items=tuple_from_df_1
            )
        )


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
        super().__init__(df_1=users_profile_df, df_2=users_rec_list_df)
        self.target_dist = None
        self.realized_dist = None

        self.items_df = items_set_df
        self._item_in_memory = None

        self.dist_func = distributions_funcs(distribution=distribution_name)
        self.dist_name = distribution_name

        self.calib_measure_func = calibration_measures_funcs(measure=distance_func_name)
        self.calib_measure_name = distance_func_name

        self.users_ix = None

    def item_preparation(self) -> None:
        """

        :return:
        """
        self._item_in_memory = ItemsInMemory(data=self.items_df)
        self._item_in_memory.item_by_genre()

    @staticmethod
    def transform_to_vec(target_dist: dict, realized_dist: dict):
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
        self.target_dist = self.compute_distribution(self.df_1)


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

    def based_on_position(self, rec_pos_df: DataFrame) -> float:
        """

        :param rec_pos_df:
        :return:
        """
        self.realized_dist = self.compute_distribution(rec_pos_df)
        results = [
            self.compute_ace(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)],
            ) for ix in self.users_ix
        ]
        return mean(results)

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        list_size = self.df_2["ORDER"].max()

        self.users_ix = list(self.target_dist.keys())
        results = [
            self.based_on_position(
                rec_pos_df=self.df_2[self.df_2["ORDER"] <= i].copy()
            ) for i in range(1, list_size + 1)
        ]
        return mean(results)


class Miscalibration(BaseCalibrationMetric):
    """
    Miscalibration. Metric to calibrated recommendations systems.

    Implementation based on:
    - Calibrated Recommendations - Steck (2018) - https://doi.org/10.1145/3240323.3240372

    """

    def compute_miscalibration(self, target_dist: dict, realized_dist: dict) -> float:
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
        self.realized_dist = self.compute_distribution(self.df_2)

        self.users_ix = list(self.target_dist.keys())

        results = [
            self.compute_miscalibration(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)]
            )
            for ix in self.users_ix
        ]

        return mean(results)


class MeanAverageMiscalibration(Miscalibration):
    """
    Mean Average Miscalibration. Metric to calibrated recommendations systems.

    Implementation based on:
    -

    """

    def based_on_position(self, rec_pos_df: DataFrame) -> float:
        """

        :param rec_pos_df:
        :return:
        """
        self.realized_dist = self.compute_distribution(rec_pos_df)
        results = [
            self.compute_miscalibration(
                self.target_dist[str(ix)],
                self.realized_dist[str(ix)],
            ) for ix in self.users_ix
        ]
        return mean(results)

    def compute(self) -> float:
        """

        :return: A float which comprises the metric value.
        """
        super().compute()
        list_size = self.df_2["ORDER"].max()

        self.users_ix = list(self.target_dist.keys())
        results = [
            self.based_on_position(
                rec_pos_df=self.df_2[self.df_2["ORDER"] <= i].copy()
            ) for i in range(1, list_size + 1)
        ]
        return mean(results)


# ################################################################################################ #
# ################################## Unexpectedness Base Metrics ################################# #
# ################################################################################################ #
class Serendipity(BaseMetric):
    """
    Serendipity.
    """

    def __init__(
            self,
            users_rec_list_df: DataFrame, users_test_df: DataFrame, users_baseline_df: DataFrame
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_test_df: A Pandas DataFrame,
            which represents the test items.
        """
        super().__init__(
            df_1=users_test_df, df_2=users_rec_list_df,
            df_3=users_baseline_df
        )

    def single_process_serend(
            self, tuple_from_df_3: tuple, tuple_from_df_2: tuple, tuple_from_df_1: tuple
    ) -> float:
        """
        This method process the metric (Serendipity) value for one user.

        :param tuple_from_df_3: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (serendipity) value for one user.
        """

        rec_items_ids = tuple_from_df_2[1]['ITEM_ID'].tolist()
        test_items_ids = tuple_from_df_1[1]['ITEM_ID'].tolist()
        baselines_items_ids = tuple_from_df_3[1]['ITEM_ID'].tolist()

        useful = list(set(rec_items_ids) & set(test_items_ids))

        unexpected_ids = list(set(rec_items_ids) - set(baselines_items_ids))

        sen = list(set(unexpected_ids) & set(useful))

        n_unexpected = 0
        if len(sen) > 0 and len(rec_items_ids) > 0:
            n_unexpected = len(sen) / len(rec_items_ids)
        return n_unexpected

    def compute(self) -> float:
        """

        :return: A float which comprises the metric value.
        """
        self.checking_users()
        self.ordering_and_grouping()

        users_results = list(map(
            self.single_process_serend,
            self.grouped_df_3,
            self.grouped_df_2,
            self.grouped_df_1
        ))
        return mean(users_results)


class Unexpectedness(BaseMetric):
    """
    Unexpectedness.
    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_df: DataFrame):
        """

        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_test_df: A Pandas DataFrame,
            which represents the test items.
        """
        super().__init__(df_1=users_test_df, df_2=users_rec_list_df)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (unexpectedness) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (unexpectedness) value for one user.
        """
        rec_items_ids = tuple_from_df_2[1]['ITEM_ID'].tolist()
        test_items_ids = tuple_from_df_1[1]['ITEM_ID'].tolist()

        unexpected_ids = list(set(rec_items_ids) - set(test_items_ids))
        n_unexpected = len(unexpected_ids) / len(rec_items_ids)
        return n_unexpected


# ################################################################################################ #
# ###################################### Verification Metrics #################################### #
# ################################################################################################ #
class AverageNumberOfOItemsChanges(BaseMetric):
    """
    Average Number of Items Changes (ANIC). A metric to get the average number of changes
        between the candidate items and recommendation list.
    """

    def __init__(self, users_rec_list_df: DataFrame, users_cand_items_df: DataFrame):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_cand_items_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(df_1=users_cand_items_df, df_2=users_rec_list_df)

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        In special, this method overrides the original one, including one more attribute to order.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'TRANSACTION_VALUE'], inplace=True, ascending=False
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (ANIC) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (ANIC) value for one user.
        """
        n = tuple_from_df_2[1]["ORDER"].max()
        set_a = tuple_from_df_1[1]["ITEM_ID"].head(n).tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()
        size = len(set(set_b) - set(set_a))
        return size


class AverageNumberOfGenreChanges(BaseMetric):
    """
    Average number of changes. A metric to get the average number of changes
        between the candidate items and recommendation list.
    """

    def __init__(
            self, users_rec_list_df: DataFrame, users_cand_items_df: DataFrame, items_df: DataFrame
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_cand_items_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(df_1=users_cand_items_df, df_2=users_rec_list_df)
        self.items_df = items_df

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        In special, this method overrides the original one, including one more attribute to order.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'TRANSACTION_VALUE'], inplace=True, ascending=False
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (ANGC) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (ANGC) value for one user.
        """
        n = tuple_from_df_2[1]["ORDER"].max()

        set_a = tuple_from_df_1[1]["ITEM_ID"].head(n).tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()

        genres_a = list(itertools.chain.from_iterable([
            genres.split("|")
            for genres in self.items_df[self.items_df["ITEM_ID"].isin(set_a)]["GENRES"].tolist()
        ]))
        genres_b = list(itertools.chain.from_iterable([
            genres.split("|")
            for genres in self.items_df[self.items_df["ITEM_ID"].isin(set_b)]["GENRES"].tolist()
        ]))

        size = len(set(genres_b) - set(genres_a))
        return size
