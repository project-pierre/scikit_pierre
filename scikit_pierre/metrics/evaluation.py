"""
This file contains all evaluation metrics.
"""
import itertools

from numpy import mean
from pandas import DataFrame

from .base import BaseMetric, BaseCalibrationMetric
from ..distributions.compute_tilde_q import compute_tilde_q


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

    def user_association_miscalibration(self, distri: dict):
        return {
            str(ix): self.compute_miscalibration(
                self.target_dist[str(ix)],
                distri[str(ix)]
            )
            for ix in self.users_ix
        }

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


class IncreaseAndDecreaseMiscalibration(Miscalibration):
    """

    """

    def __init__(
            self,
            users_profile_df: DataFrame, users_rec_list_df: DataFrame,
            users_baseline_df: DataFrame, items_df: DataFrame,
            distribution_name: str = "CWS", distance_func_name: str = "KL"
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_baseline_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(
            users_profile_df=users_profile_df, users_rec_list_df=users_rec_list_df,
            items_set_df=items_df, distribution_name=distribution_name,
            distance_func_name=distance_func_name
        )
        self.df_3 = users_baseline_df
        self.distri_df_3 = None
        self.increase = True

    def set_choice(self, choice: bool) -> None:
        self.increase = choice

    def count(self) -> float:
        rec_dist = self.user_association_miscalibration(
            self.realized_dist
        )
        base_dist = self.user_association_miscalibration(
            self.distri_df_3
        )
        if self.increase:
            return len([
                ix
                for ix in self.users_ix
                if rec_dist[ix] >= base_dist[ix]
            ])
        else:
            return len([
                ix
                for ix in self.users_ix
                if rec_dist[ix] < base_dist[ix]
            ])

    def compute(self) -> float:
        """

        :return:
        """
        self.checking_users()
        self.target_dist = self.compute_distribution(self.df_1)
        self.realized_dist = self.compute_distribution(self.df_2)
        self.distri_df_3 = self.compute_distribution(self.df_3)

        self.users_ix = list(self.target_dist.keys())

        return self.count()


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

    def __init__(self, users_rec_list_df: DataFrame, users_baseline_df: DataFrame):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_baseline_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(df_1=users_baseline_df, df_2=users_rec_list_df)

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        In special, this method overrides the original one, including one more attribute to order.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (ANIC) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (ANIC) value for one user.
        """
        set_a = tuple_from_df_1[1]["ITEM_ID"].tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()
        size = len(set(set_b) - set(set_a))
        return size


class AverageNumberOfGenreChanges(BaseMetric):
    """
    Average number of changes. A metric to get the average number of changes
    between the candidate items and recommendation list.
    """

    def __init__(
            self, users_rec_list_df: DataFrame, users_baseline_df: DataFrame, items_df: DataFrame
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_baseline_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(df_1=users_baseline_df, df_2=users_rec_list_df)
        self.items_df = items_df

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        In special, this method overrides the original one, including one more attribute to order.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        This method process the metric (ANGC) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_1: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (ANGC) value for one user.
        """

        set_a = tuple_from_df_1[1]["ITEM_ID"].tolist()
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


class ExplainingMiscalibration(BaseCalibrationMetric):
    """
    Explaining Miscalibration. Metric to explain the recommendations Based on calibration.

    Implementation based on:
    -

    """

    def __init__(
            self,
            users_profile_df: DataFrame, users_rec_list_df: DataFrame,
            users_baseline_df: DataFrame, items_df: DataFrame,
            distribution_name: str = "CWS", distance_func_name: str = "KL"
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.

        :param users_baseline_df: A Pandas DataFrame,
            which represents the candidate items.
        """
        super().__init__(
            users_profile_df=users_profile_df, users_rec_list_df=users_rec_list_df,
            items_set_df=items_df, distribution_name=distribution_name,
            distance_func_name=distance_func_name
        )
        self.df_3 = users_baseline_df
        self.distri_df_3 = None

    def ordering(self) -> None:
        """
        This method is to order the Dataframe based on the user ids.
        In special, this method overrides the original one, including one more attribute to order.
        """
        self.df_3.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )
        self.df_2.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )

    @staticmethod
    def single_process_anic(tuple_from_df_2: tuple, tuple_from_df_3: tuple) -> float:
        """
        This method process the metric (ANIC) value for one user.

        :param tuple_from_df_2: A tuple where: 0 is the user id and 1 is a Dataframe.
        :param tuple_from_df_3: A tuple where: 0 is the user id and 1 is a Dataframe.

        :return: A float which comprises the metric (ANIC) value for one user.
        """
        set_a = tuple_from_df_3[1]["ITEM_ID"].tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()
        size = len(set(set_b) - set(set_a))
        return size

    def find_user_based_on_changes(self) -> dict:
        """

        :return:
        """
        return {
            str(g2[0][0]): self.single_process_anic(
                tuple_from_df_2=g2,
                tuple_from_df_3=g3
            )
            for g2, g3 in zip(
                self.grouped_df_2,
                self.grouped_df_3
            )
        }

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

    def user_association_miscalibration(self, distri: dict):
        return {
            str(ix): self.compute_miscalibration(
                self.target_dist[str(ix)],
                distri[str(ix)]
            )
            for ix in self.users_ix
        }

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        self.ordering_and_grouping()

        self.users_ix = list(self.target_dist.keys())

        self.realized_dist = self.compute_distribution(self.df_2)
        mis_2_results = self.user_association_miscalibration(
            distri=self.realized_dist
        )

        self.distri_df_3 = self.compute_distribution(self.df_3)
        mis_3_results = self.user_association_miscalibration(
            distri=self.distri_df_3
        )

        anic_results = self.find_user_based_on_changes()

        _min_value = min(anic_results.values())
        _max_value = max(anic_results.values())

        _min_changes_high = []
        _max_changes_high = []

        _min_changes_lower = []
        _max_changes_lower = []

        _min_calib = []
        _max_calib = []

        _aux_min = 1000
        _aux_id_min = 1000
        _aux_max = 0
        _aux_id_max = 0
        for _ix in anic_results.keys():
            if mis_2_results[str(_ix)] < _aux_min:
                _aux_min = mis_2_results[str(_ix)]
                _aux_id_min = str(_ix)

            if mis_2_results[str(_ix)] > _aux_max:
                _aux_max = mis_2_results[str(_ix)]
                _aux_id_max = str(_ix)

            if (anic_results[str(_ix)] == _min_value and
                    mis_2_results[str(_ix)] > mis_3_results[str(_ix)]):
                _min_changes_high.append(str(_ix))

            if (anic_results[str(_ix)] == _max_value and
                    mis_2_results[str(_ix)] > mis_3_results[str(_ix)]):
                _max_changes_high.append(str(_ix))

            if (anic_results[str(_ix)] == _min_value and
                    mis_2_results[str(_ix)] < mis_3_results[str(_ix)]):
                _min_changes_lower.append(str(_ix))

            if (anic_results[str(_ix)] == _max_value and
                    mis_2_results[str(_ix)] < mis_3_results[str(_ix)]):
                _max_changes_lower.append(str(_ix))

        if len(_min_changes_lower) > 0:
            self.printing_list_changing(
                user_id=_min_changes_lower[0],
                calib_base=mis_3_results[str(_min_changes_lower[0])],
                calib_rec=mis_2_results[str(_min_changes_lower[0])]
            )

        if len(_min_changes_high) > 0:
            self.printing_list_changing(
                user_id=_min_changes_high[0],
                calib_base=mis_3_results[str(_min_changes_high[0])],
                calib_rec=mis_2_results[str(_min_changes_high[0])]
            )

        self.printing_list_changing(
            user_id=_aux_id_min,
            calib_base=mis_3_results[str(_aux_id_min)],
            calib_rec=mis_2_results[str(_aux_id_min)]
        )

        self.printing_list_changing(
            user_id=_aux_id_max,
            calib_base=mis_3_results[str(_aux_id_max)],
            calib_rec=mis_2_results[str(_aux_id_max)]
        )

        return 0.0

    def printing_list_changing(self, user_id: str, calib_base, calib_rec):
        user_rec_ids = self.df_2[self.df_2["USER_ID"] == int(user_id)]["ITEM_ID"].tolist()
        user_base_ids = self.df_3[self.df_3["USER_ID"] == int(user_id)]["ITEM_ID"].tolist()

        rec_changed = list(set(user_rec_ids) - set(user_base_ids))
        base_changed = list(set(user_base_ids) - set(user_rec_ids))

        rec_list = self.items_df[self.items_df["ITEM_ID"].isin(rec_changed)]
        base_list = self.items_df[self.items_df["ITEM_ID"].isin(base_changed)]

        genres_a = list(itertools.chain.from_iterable([
            genres.split("|")
            for genres in rec_list["GENRES"].tolist()
        ]))

        genres_b = list(itertools.chain.from_iterable([
            genres.split("|")
            for genres in base_list["GENRES"].tolist()
        ]))

        print("\n")

        print("-" * 100)
        print(
            "User: ", user_id, " - ",
            "The miscalibration goes from: ", calib_base, " To ", calib_rec
        )

        print("Item included in the recommendation list: ", len(rec_list))
        print(rec_list)

        print("\n")

        print("Item excluded from the recommendation list: ", len(base_list))
        print(base_list)

        print("-" * 100)

        rec_genres = list(set(genres_a) - set(genres_b))
        print("Genres included in the recommendation list: ", len(rec_genres))
        print(rec_genres)

        print("\n")

        base_genres = list(set(genres_b) - set(genres_a))
        print("Genres excluded from the recommendation list: ", len(base_genres))
        print(base_genres)

        print("-" * 100)

        print("\n")
