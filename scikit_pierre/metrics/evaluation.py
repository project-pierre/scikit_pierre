"""
Evaluation metrics for recommendation systems.

Provides accuracy, diversity, calibration, and unexpectedness metrics,
all operating on Pandas DataFrames that represent users' recommendation
lists, test-item sets, and interaction profiles.

Classes
-------
MeanAveragePrecision (MAP)
    Ranking accuracy metric.
MeanReciprocalRank (MRR)
    Accuracy metric based on the first relevant item's rank.
IntraListSimilarity (ILS)
    Intra-list diversity based on cosine similarity.
Personalization
    Cross-user diversity based on cosine similarity of recommendation sets.
Novelty
    Self-information-based novelty.
Coverage
    Catalogue coverage (percentage of unique items recommended).
MeanAbsoluteCalibrationError (MACE)
    Position-aware calibration error.
Miscalibration
    KL-divergence-based calibration metric (Steck, 2018).
MeanAverageMiscalibration
    Position-averaged miscalibration.
NumberOfUserIncreaseAndDecreaseMiscalibration
    Count of users whose miscalibration changed relative to a baseline.
UserIDMiscalibration
    Mean miscalibration value among users with changed miscalibration.
Serendipity
    Unexpectedly relevant items.
Unexpectedness
    Items not present in the test set.
AverageNumberOfOItemsChanges (ANIC)
    Mean item-level changes relative to a baseline list.
AverageNumberOfGenreChanges (ANGC)
    Mean genre-level changes relative to a baseline list.
"""
# pylint: disable=too-many-lines
import itertools
from collections import Counter
from typing import List

from numpy import mean, triu_indices, array, log2, sum as np_sum, linalg
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
    Intra-List Similarity (ILS) diversity metric.

    Measures the average pairwise cosine similarity between items in each
    user's recommendation list.  Lower values indicate higher diversity.

    Parameters are set via the constructor; call :meth:`compute` to obtain
    the aggregated ILS value across all users.
    """

    def __init__(
            self, users_rec_list_df: DataFrame, items_df: DataFrame, encoded_df: DataFrame = None
    ):
        """
        Parameters
        ----------
        users_rec_list_df : DataFrame
            Recommendation lists with ``USER_ID`` and ``ITEM_ID`` columns.
        items_df : DataFrame
            Item catalogue with ``ITEM_ID`` and ``GENRES`` columns.
        encoded_df : DataFrame, optional
            Pre-computed one-hot encoded item feature matrix.  When
            ``None``, it is computed lazily by :meth:`encoding`.
        """
        self.rec_list_df = users_rec_list_df
        self.items = items_df
        self.encoded = encoded_df

    def encoding(self):
        """
        Build the one-hot genre encoding if it has not been provided.

        Stores the result in ``self.encoded`` for reuse.
        """
        if self.encoded is None:
            _items = ItemsInMemory(data=self.items)
            _items.one_hot_encode()
            self.encoded = _items.get_encoded()

    def compute(self):
        """
        Compute the mean Intra-List Similarity across all users.

        Returns
        -------
        float
            Average ILS value (lower = more diverse).
        """
        rec_set = [
            row["ITEM_ID"].tolist() for ix, row in
            self.rec_list_df.groupby(by=["USER_ID"])
        ]
        self.encoding()

        ils = [self._single_list_similarity(u_rec) for u_rec in rec_set]
        return mean(ils)

    def _single_list_similarity(self, predicted: list) -> float:
        """
        Compute the Intra-List Similarity for one user's recommendation list.

        Parameters
        ----------
        predicted : list
            Ordered item IDs in the recommendation list, e.g.
            ``['X', 'Y', 'Z']``.  Each ID must be present as an index
            in ``self.encoded``.

        Returns
        -------
        float
            Mean pairwise cosine similarity over the upper triangle of
            the item-by-item similarity matrix (diagonal excluded).
        """
        vecs = self.encoded.loc[predicted].fillna(0).values.astype(float)

        # Normalize rows to unit length for cosine similarity via dot product.
        norms = linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms

        similarity = vecs @ vecs.T

        # Upper triangle (k=1) excludes self-similarity on the diagonal.
        upper_right = triu_indices(similarity.shape[0], k=1)

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
        Build a sparse user-item binary matrix from the recommendation lists.

        Each row is a user; each column is an item; a cell is 1 when the
        item appears in that user's recommendation list, 0 otherwise.

        Returns
        -------
        scipy.sparse.csr_matrix
            Binary user-item recommendation matrix.
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
        Compute the Personalization score across all users.

        Returns
        -------
        float
            ``1 - mean_pairwise_cosine_similarity`` across all user pairs.
            Values close to 1.0 indicate high personalisation (diverse lists
            across users); values close to 0.0 indicate most users received
            the same items.
        """
        rec_matrix_sparse = self.make_rec_matrix()

        similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

        # Subtract the diagonal (self-similarity = 1) before averaging.
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
    def single_process_nov(predicted: List[list], pop: dict, u: int, n: int) -> float:
        """
        Compute the Novelty score given pre-aggregated popularity counts.

        Parameters
        ----------
        predicted : list of list
            Per-user recommendation lists (each inner list contains item IDs).
        pop : dict
            Mapping of item_id -> interaction count across all users.
        u : int
            Total number of unique users in the training data.
        n : int
            Recommendation list length (maximum ORDER value).

        Returns
        -------
        float
            Mean self-information across all users and positions.
        """

        mean_self_information = []
        k = 0
        for sublist in predicted:
            self_information = 0
            k += 1

            for i in sublist:
                v = pop[str(i)]
                if v == 0:
                    v = 0.00001
                self_information += np_sum(-log2(v/u))
            mean_self_information.append(self_information/n)
        novelty = np_sum(mean_self_information)/k
        return novelty

    def compute(self):
        """

        :return:
        """

        rec_set = [row["ITEM_ID"].tolist() for ix, row in self.df_2.groupby(by=["USER_ID"])]
        pop = dict(Counter(self.df_1["ITEM_ID"].tolist()))
        item_ids = self.items_df["ITEM_ID"].tolist()
        diff_ids = set(item_ids) - set(pop.keys())
        for diff_id in diff_ids:
            pop[diff_id] = 0
        u = self.df_1["USER_ID"].nunique()

        return self.single_process_nov(
            predicted=rec_set, pop=pop, u=u, n=max(self.df_2["ORDER"])
        )


class Coverage(BaseMetric):
    """
    Catalogue Coverage metric.

    Measures the percentage of the item catalogue that appears in at least
    one user's recommendation list.
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
        Compute the Absolute Calibration Error for one user at one position.

        Parameters
        ----------
        target_dist : dict
            Target genre distribution *p*.
        realized_dist : dict
            Realized genre distribution *q* for the truncated list.

        Returns
        -------
        float
            Mean absolute element-wise difference between *p* and *q*.
        """
        p, q = self.transform_to_vec(target_dist, realized_dist)
        diff_result = [abs(t_value - r_value) for t_value, r_value in zip(p, q)]
        return mean(diff_result)

    def based_on_position(self, rec_pos_df: DataFrame) -> float:
        """
        Compute MACE for all users given a recommendation list truncated to a
        specific position.

        Parameters
        ----------
        rec_pos_df : DataFrame
            Recommendation list filtered to ``ORDER <= k`` for some position
            *k*.

        Returns
        -------
        float
            Mean ACE across all users at position *k*.
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
        Compute the Mean Absolute Calibration Error (MACE) averaged over all
        list positions from 1 to ``list_size``.

        Returns
        -------
        float
            Position-averaged MACE value.
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
        Compute the miscalibration between a target and a realized distribution.

        The realized distribution is first smoothed via tilde-q before
        the configured divergence measure is applied.

        Parameters
        ----------
        target_dist : dict
            Target genre distribution *p*.
        realized_dist : dict
            Realized genre distribution *q*.

        Returns
        -------
        float
            Divergence (or similarity) value for this user.
        """
        p, q = self.transform_to_vec(target_dist, realized_dist)
        return self.calib_measure_func(
            p=p,
            q=compute_tilde_q(p=p, q=q)
        )

    def user_association_miscalibration(self, distri: dict):
        """Map each user id to their miscalibration score."""
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
    Mean Average Miscalibration (MAM).

    Extends :class:`Miscalibration` by averaging miscalibration values
    across all list positions (1 to ``list_size``), giving a position-aware
    calibration score that penalises miscalibration appearing early in the
    list more than late occurrences.
    """

    def based_on_position(self, rec_pos_df: DataFrame) -> float:
        """
        Compute mean miscalibration across users for a truncated list.

        Parameters
        ----------
        rec_pos_df : DataFrame
            Recommendation list filtered to ``ORDER <= k``.

        Returns
        -------
        float
            Mean miscalibration across all users at position *k*.
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
    Count users whose miscalibration increased or decreased relative to a baseline.

    Compares the miscalibration of the recommendation list against a
    baseline (e.g. the original candidate list) and returns the count of
    users for whom the miscalibration either increased (``increase=True``)
    or decreased (``increase=False``).
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self,
            users_profile_df: DataFrame, users_rec_list_df: DataFrame,
            users_baseline_df: DataFrame, items_df: DataFrame,
            distribution_name: str = "CWS", distance_func_name: str = "KL"
    ):
        """
        Parameters
        ----------
        users_profile_df : DataFrame
            User interaction history for computing the target distribution.
        users_rec_list_df : DataFrame
            Recommendation lists to evaluate.
        users_baseline_df : DataFrame
            Baseline (e.g. candidate) recommendation lists used for comparison.
        items_df : DataFrame
            Item catalogue with ``ITEM_ID`` and ``GENRES``.
        distribution_name : str, optional
            Genre distribution strategy acronym.  Defaults to ``"CWS"``.
        distance_func_name : str, optional
            Calibration measure acronym.  Defaults to ``"KL"``.
        """
        super().__init__(
            users_profile_df=users_profile_df, users_rec_list_df=users_rec_list_df,
            items_set_df=items_df, distribution_name=distribution_name,
            distance_func_name=distance_func_name
        )
        self.df_3 = users_baseline_df
        self.distri_df_3 = None
        # Default: count users whose miscalibration *increased*.
        self.increase = True
        self.with_profile = True

    def set_choice(self, choice: bool) -> None:
        """
        Set whether to count users with *increased* or *decreased* miscalibration.

        Parameters
        ----------
        choice : bool
            ``True`` to count users with increased miscalibration (default);
            ``False`` to count users with decreased miscalibration.
        """
        self.increase = choice

    def set_comparison(self, choice: bool) -> None:
        """
        Set whether the baseline comparison uses the user profile or the candidate list.

        Parameters
        ----------
        choice : bool
            ``True`` to compare against the user profile; ``False`` to
            compare against the candidate list.
        """
        self.with_profile = choice

    def selecting_users(self) -> list:
        """
        Return miscalibration values for users whose miscalibration changed
        in the configured direction relative to the baseline.

        Returns
        -------
        list of float
            Miscalibration values for matching users.
        """
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
        return [
            rec_miscalib[ix]
            for ix in self.users_ix
            if rec_miscalib[ix] < base_miscalib[ix]
        ]

    def base_dist_compute(self):
        """
        Pre-compute target, realized, and baseline distributions.

        Validates user consistency, then populates ``self.target_dist``,
        ``self.realized_dist``, ``self.distri_df_3``, and ``self.users_ix``
        as a shared setup step used by :meth:`compute`.
        """
        self.checking_users()
        self.compute_target_dist()
        self.compute_realized_dist()
        self.distri_df_3 = self.compute_distribution(self.df_3)
        self.users_ix = list(self.target_dist.keys())

    def compute(self) -> float:
        """
        Return the number of users whose miscalibration changed in the
        configured direction.

        Returns
        -------
        float
            Count of matching users (returned as float for API consistency).
        """
        self.base_dist_compute()

        return len(self.selecting_users())


class UserIDMiscalibration(NumberOfUserIncreaseAndDecreaseMiscalibration):
    """
    Mean miscalibration value among users whose calibration changed.

    Extends :class:`NumberOfUserIncreaseAndDecreaseMiscalibration` by
    returning the *mean* miscalibration of the affected users rather than
    the count.
    """

    def selecting_values(self):
        """
        Return miscalibration values for the selected users.

        Returns
        -------
        list of float
            Miscalibration values from :meth:`selecting_users`.
        """
        return list(self.selecting_users())

    def compute(self) -> float:
        """
        Compute the mean miscalibration of users whose calibration changed.

        Returns
        -------
        float
            Mean miscalibration value across matching users.
        """
        self.base_dist_compute()

        return mean(self.selecting_values())


# ################################################################################################ #
# ################################## Unexpectedness Base Metrics ################################# #
# ################################################################################################ #
class Serendipity(BaseMetric):
    """
    Serendipity metric.

    Measures the proportion of recommended items that are both relevant
    (present in the test set) and unexpected (absent from the baseline list).
    """

    def __init__(
            self,
            users_rec_list_df: DataFrame, users_test_df: DataFrame, users_baseline_df: DataFrame
    ):
        """
        Parameters
        ----------
        users_rec_list_df : DataFrame
            Recommendation lists with ``USER_ID`` and ``ITEM_ID``.
        users_test_df : DataFrame
            Held-out test items used to determine relevance.
        users_baseline_df : DataFrame
            Baseline (e.g. popularity-based) lists used to determine
            unexpectedness.
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
        Compute the Serendipity score for a single user.

        An item is serendipitous if it is both relevant (in the test set)
        and unexpected (not in the baseline list).

        Parameters
        ----------
        tuple_from_df_3 : tuple
            ``(user_id, DataFrame)`` for the baseline list.
        tuple_from_df_2 : tuple
            ``(user_id, DataFrame)`` for the recommendation list.
        tuple_from_df_1 : tuple
            ``(user_id, DataFrame)`` for the test-item set.

        Returns
        -------
        float
            Fraction of recommended items that are serendipitous.
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
        Compute the mean Serendipity score across all users.

        Returns
        -------
        float
            Mean fraction of serendipitous items across all users.
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
    Unexpectedness metric.

    Measures the fraction of recommended items that do *not* appear in the
    user's test set, capturing how surprising the recommendation list is.
    """

    def __init__(self, users_rec_list_df: DataFrame, users_test_df: DataFrame):
        """
        Parameters
        ----------
        users_rec_list_df : DataFrame
            Recommendation lists with ``USER_ID`` and ``ITEM_ID``.
        users_test_df : DataFrame
            Held-out test items.
        """
        super().__init__(df_1=users_test_df, df_2=users_rec_list_df)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        Compute the Unexpectedness score for a single user.

        Parameters
        ----------
        tuple_from_df_2 : tuple
            ``(user_id, DataFrame)`` for the recommendation list.
        tuple_from_df_1 : tuple
            ``(user_id, DataFrame)`` for the test-item set.

        Returns
        -------
        float
            Fraction of recommended items not present in the test set.
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
        Parameters
        ----------
        users_rec_list_df : DataFrame
            Calibrated recommendation lists with ``USER_ID``, ``ITEM_ID``,
            and ``ORDER``.
        users_baseline_df : DataFrame
            Baseline (candidate) lists with ``USER_ID``, ``ITEM_ID``, and
            ``ORDER``.
        """
        super().__init__(df_1=users_baseline_df, df_2=users_rec_list_df)

    def ordering(self) -> None:
        """
        Sort both DataFrames by ``USER_ID`` then ``ORDER``.

        Overrides the base implementation to preserve position ordering
        needed for item-level change detection.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        Compute the ANIC score for a single user.

        Parameters
        ----------
        tuple_from_df_2 : tuple
            ``(user_id, DataFrame)`` for the recommendation list.
        tuple_from_df_1 : tuple
            ``(user_id, DataFrame)`` for the baseline list.

        Returns
        -------
        float
            Number of items present in the recommendation list but absent
            from the baseline list.
        """
        set_a = tuple_from_df_1[1]["ITEM_ID"].tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()
        size = len(set(set_b) - set(set_a))
        return size


class AverageNumberOfGenreChanges(BaseMetric):
    """
    Average Number of Genre Changes (ANGC).

    Counts how many new genres appear in the recommendation list that were
    not present in the baseline (candidate) list.  This metric measures
    genre-level diversity introduced by calibration.
    """

    def __init__(
            self, users_rec_list_df: DataFrame, users_baseline_df: DataFrame, items_df: DataFrame
    ):
        """
        Parameters
        ----------
        users_rec_list_df : DataFrame
            Calibrated recommendation lists with ``USER_ID``, ``ITEM_ID``,
            and ``ORDER``.
        users_baseline_df : DataFrame
            Baseline lists with ``USER_ID``, ``ITEM_ID``, and ``ORDER``.
        items_df : DataFrame
            Item catalogue with ``ITEM_ID`` and ``GENRES`` (pipe-separated).
        """
        super().__init__(df_1=users_baseline_df, df_2=users_rec_list_df)
        self.items_df = items_df
        self._genre_lookup = {
            str(item_id): genres_str.split("|")
            for item_id, genres_str in zip(items_df["ITEM_ID"], items_df["GENRES"])
        }

    def ordering(self) -> None:
        """
        Sort both DataFrames by ``USER_ID`` then ``ORDER``.

        Overrides the base implementation to preserve position ordering.
        """
        self.df_1.sort_values(
            by=['USER_ID', 'ORDER'], inplace=True
        )
        self.df_2.sort_values(by=['USER_ID', 'ORDER'], inplace=True)

    def single_process(self, tuple_from_df_2: tuple, tuple_from_df_1: tuple) -> float:
        """
        Compute the ANGC score for a single user.

        Parameters
        ----------
        tuple_from_df_2 : tuple
            ``(user_id, DataFrame)`` for the recommendation list.
        tuple_from_df_1 : tuple
            ``(user_id, DataFrame)`` for the baseline list.

        Returns
        -------
        float
            Number of unique genres present in the recommendation list's
            items but absent from the baseline list's items.
        """

        set_a = tuple_from_df_1[1]["ITEM_ID"].tolist()
        set_b = tuple_from_df_2[1]["ITEM_ID"].tolist()

        genres_a = list(itertools.chain.from_iterable(
            self._genre_lookup[str(i)] for i in set_a if str(i) in self._genre_lookup
        ))
        genres_b = list(itertools.chain.from_iterable(
            self._genre_lookup[str(i)] for i in set_b if str(i) in self._genre_lookup
        ))

        size = len(set(genres_b) - set(genres_a))
        return size
