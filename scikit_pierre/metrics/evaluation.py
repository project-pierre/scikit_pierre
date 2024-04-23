"""
This file contains all evaluation metrics.
"""
import itertools
from collections import Counter
from typing import List

from numpy import mean, triu_indices, array, log2
from pandas import DataFrame, notna
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseMetric, BaseCalibrationMetric
from ..distributions.compute_tilde_q import compute_tilde_q
from ..models.item import ItemsInMemory


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
# ####################################### Diversity Metrics ###################################### #
# ################################################################################################ #
class IntraListSimilarity:
    """

    """

    def __init__(
            self, users_rec_list_df: DataFrame, items_df: DataFrame, encoded_df: DataFrame = None
    ):
        self.rec_list_df = users_rec_list_df
        self.items = items_df
        self.encoded = encoded_df

    def encoding(self):
        if self.encoded is None:
            _items = ItemsInMemory(data=self.items)
            _items.one_hot_encode()
            self.encoded = _items.get_encoded()

    def compute(self):
        rec_set = [
            row["ITEM_ID"].tolist() for ix, row in
            self.rec_list_df.groupby(by=["USER_ID"])
        ]
        self.encoding()

        ils = [self._single_list_similarity(u_rec) for u_rec in rec_set]
        return mean(ils)

    def _single_list_similarity(self, predicted: list) -> float:
        """
        Computes the intra-list similarity for a single list of recommendations.
        Parameters
        ----------
        predicted : a list
            Ordered predictions
            Example: ['X', 'Y', 'Z']
        feature_df: dataframe
            A dataframe with one hot encoded or latent features.
            The dataframe should be indexed by the id used in the recommendations.
        Returns:
        -------
        ils_single_user: float
            The intra-list similarity for a single list of recommendations.
        """
        # get features for all recommended items
        recs_content = self.encoded.loc[predicted]
        recs_content = recs_content.dropna()
        recs_content = sp.csr_matrix(recs_content.values)

        # calculate similarity scores for all items in list
        similarity = cosine_similarity(X=recs_content, dense_output=False)

        # get indicies for upper right triangle w/o diagonal
        upper_right = triu_indices(similarity.shape[0], k=1)

        # calculate average similarity score of all recommended items in list
        ils_single_user = mean(similarity[upper_right])
        return ils_single_user


class Personalization(BaseMetric):
    """
    Personalization.
    """

    def __init__(
            self,
            users_rec_list_df: DataFrame
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.
        """
        super().__init__(
            df_2=users_rec_list_df
        )

    def make_rec_matrix(self) -> sp.csr_matrix:
        """
        This method construct the matrix to process the personalization.
        :return:
        """
        predicted = [row["ITEM_ID"].tolist() for ix, row in self.df_2.groupby(by=["USER_ID"])]
        predicted = array(predicted)
        df = DataFrame(data=predicted).reset_index().melt(
            id_vars='index', value_name='item',
        )
        df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
        df = notna(df)*1
        rec_matrix = sp.csr_matrix(df.values)
        return rec_matrix

    def compute(self):
        """

        :return:
        """

        # create matrix for recommendations
        rec_matrix_sparse = self.make_rec_matrix()

        # calculate similarity for every user's recommendation list
        similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

        # calculate average similarity
        dim = similarity.shape[0]
        personalization = (similarity.sum() - dim) / (dim * (dim - 1))
        return 1-personalization


class Novelty(BaseMetric):
    """
    Novelty.
    """

    def __init__(
            self,
            users_profile_df: DataFrame,
            users_rec_list_df: DataFrame,
            items_df: DataFrame,
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.
        """
        super().__init__(
            df_2=users_rec_list_df,
            df_1=users_profile_df
        )
        self.items_df = items_df

    @staticmethod
    def single_process_nov(predicted: List[list], pop: dict, u: int, n: int) -> (float, list):
        """
        This method construct the matrix to process the personalization.
        :return:
        """

        mean_self_information = []
        k = 0
        for sublist in predicted:
            self_information = 0
            k += 1
            for i in sublist:
                self_information += sum(-log2(pop[i]/u))
            mean_self_information.append(self_information/n)
        novelty = sum(mean_self_information)/k
        return novelty, mean_self_information

    def compute(self):
        """

        :return:
        """

        rec_set = [row["ITEM_ID"].tolist() for ix, row in self.df_2.groupby(by=["USER_ID"])]
        pop = Counter(self.df_1["ITEM_ID"].tolist())
        u = self.df_1["USER_ID"].nunique()

        return self.single_process_nov(
            predicted=rec_set, pop=pop, u=u, n=max(self.df_2["ORDER"])
        )


class Coverage(BaseMetric):
    """
    Personalization.
    """

    def __init__(
            self,
            users_rec_list_df: DataFrame,
            items_df: DataFrame,
    ):
        """
        :param users_rec_list_df: A Pandas DataFrame,
            which represents the users recommendation lists.
        """
        super().__init__(
            df_2=users_rec_list_df
        )
        self.items_df = items_df

    def compute(self):
        """

        :return:
        """
        predicted = [row["ITEM_ID"].tolist() for ix, row in self.df_2.groupby(by=["USER_ID"])]
        catalog = self.items_df["ITEM_ID"].tolist()

        predicted_flattened = [p for sublist in predicted for p in sublist]
        unique_predictions = len(set(predicted_flattened))
        prediction_coverage = round(unique_predictions/(len(catalog) * 1.0) * 100, 2)

        return prediction_coverage


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
                self.target_dist[ix],
                self.realized_dist[ix],
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
            ix: self.compute_miscalibration(
                self.target_dist[ix],
                distri[ix]
            )
            for ix in self.users_ix
        }

    def compute(self) -> float:
        """

        :return:
        """
        super().compute()
        self.compute_realized_dist()

        self.users_ix = list(self.target_dist.keys())

        results = [
            self.compute_miscalibration(
                self.target_dist[ix],
                self.realized_dist[ix]
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
                self.target_dist[ix],
                self.realized_dist[ix],
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


class NumberOfUserIncreaseAndDecreaseMiscalibration(Miscalibration):
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
        self.with_profile = True

    def set_choice(self, choice: bool) -> None:
        self.increase = choice

    def set_comparison(self, choice: bool) -> None:
        self.with_profile = choice

    def selecting_users(self) -> list:
        rec_miscalib = self.user_association_miscalibration(
            self.realized_dist
        )
        base_miscalib = self.user_association_miscalibration(
            self.distri_df_3
        )
        if self.increase:
            return [
                rec_miscalib[ix]
                for ix in self.users_ix
                if rec_miscalib[ix] >= base_miscalib[ix]
            ]
        else:
            return [
                rec_miscalib[ix]
                for ix in self.users_ix
                if rec_miscalib[ix] < base_miscalib[ix]
            ]

    def base_dist_compute(self):
        self.checking_users()
        self.compute_target_dist()
        self.compute_realized_dist()
        self.distri_df_3 = self.compute_distribution(self.df_3)
        self.users_ix = list(self.target_dist.keys())

    def compute(self) -> float:
        """

        :return:
        """
        self.base_dist_compute()

        return len(self.selecting_users())


class UserIDMiscalibration(NumberOfUserIncreaseAndDecreaseMiscalibration):

    def selecting_values(self):
        """

        :return:
        """
        return [
            value
            for value in self.selecting_users()
        ]

    def compute(self) -> float:
        """

        :return:
        """
        self.base_dist_compute()

        return mean(self.selecting_values())


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

    @staticmethod
    def single_process_serend(
             tuple_from_df_3: tuple, tuple_from_df_2: tuple, tuple_from_df_1: tuple
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
