"""
Explainability metric for calibrated recommendations.

Provides :class:`ExplainingMiscalibration`, a diagnostic class that
identifies users with the largest and smallest item-level changes between
a baseline and a calibrated recommendation list and prints a
position-by-position explanation of how each item swap affected
miscalibration.
"""

import itertools

from pandas import DataFrame

from ..distributions.compute_tilde_q import compute_tilde_q
from ..metrics.base import BaseCalibrationMetric


class ExplainingMiscalibration(BaseCalibrationMetric):
    """
    Diagnostic tool for explaining recommendation changes in terms of calibration.

    Computes miscalibration for both the recommendation list and a baseline
    list, identifies users with the fewest and most item changes (ANIC), and
    prints a position-by-position explanation of how each item swap affected
    calibration.

    Notes
    -----
    This class is primarily intended for qualitative analysis and prints
    results directly to stdout via :meth:`printing_list_changing` and
    :meth:`user_explain_history`.  Its :meth:`compute` method always returns
    0.0 as a sentinel value.
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
            Calibrated recommendation lists to be explained.
        users_baseline_df : DataFrame
            Baseline (e.g. candidate) lists for comparison.
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
        Compute the number of item changes (ANIC) per user between the
        recommendation list and the baseline list.

        Returns
        -------
        dict
            Mapping of str(user_id) -> number of items changed.
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
        Compute miscalibration between a target and a realized distribution.

        Parameters
        ----------
        target_dist : dict
            Target genre distribution *p*.
        realized_dist : dict
            Realized genre distribution *q*.

        Returns
        -------
        float
            Divergence value after tilde-q smoothing.
        """
        p, q = self.transform_to_vec(target_dist, realized_dist)
        return self.calib_measure_func(
            p=p,
            q=compute_tilde_q(p=p, q=q)
        )

    def user_association_miscalibration(self, distri: dict):
        """Map each user id to their miscalibration score."""
        return {
            str(ix): self.compute_miscalibration(
                self.target_dist[ix],
                distri[ix]
            )
            for ix in self.users_ix
        }

    def compute(self) -> float:  # pylint: disable=too-many-locals
        """
        Run the explanation pipeline and print results to stdout.

        Identifies extreme users (min/max ANIC) and users whose calibration
        improved or worsened, then prints a per-position change analysis for
        selected users.

        Returns
        -------
        float
            Always returns 0.0; outputs are printed to stdout.
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
            self.user_explain_history(user_id=_aux_id_min)
            self.printing_list_changing(
                user_id=_min_changes_lower[0],
                calib_base=mis_3_results[str(_min_changes_lower[0])],
                calib_rec=mis_2_results[str(_min_changes_lower[0])]
            )
        #
        if len(_min_changes_high) > 0:
            self.user_explain_history(user_id=_min_changes_high[0])
            self.printing_list_changing(
                user_id=_min_changes_high[0],
                calib_base=mis_3_results[str(_min_changes_high[0])],
                calib_rec=mis_2_results[str(_min_changes_high[0])]
            )

        return 0.0

    def printing_list_changing(self, user_id: str, calib_base: float, calib_rec: float):
        """
        Print a summary of item and genre changes for a single user.

        Shows which items were added to or removed from the recommendation
        list relative to the baseline, along with the before/after
        miscalibration values.

        Parameters
        ----------
        user_id : str
            User identifier.
        calib_base : float
            Miscalibration of the baseline list.
        calib_rec : float
            Miscalibration of the recommendation list.
        """
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

        self.user_analyzing_genres(genres_a, genres_b)

        print("-" * 100)

        print("\n")

    @staticmethod
    def user_analyzing_genres(genres_a: list, genres_b: list):
        """
        Print genre-level inclusion and exclusion summary.

        Parameters
        ----------
        genres_a : list of str
            Genres present in the recommendation list (added items).
        genres_b : list of str
            Genres present in the baseline list (removed items).
        """
        rec_genres = list(set(genres_a) - set(genres_b))
        print("Genres included in the recommendation list: ", len(rec_genres))
        print(rec_genres)

        print("\n")

        base_genres = list(set(genres_b) - set(genres_a))
        print("Genres excluded from the recommendation list: ", len(base_genres))
        print(base_genres)

    def user_explain_history(self, user_id: str):
        """
        Print a position-by-position explanation of how item swaps affected
        miscalibration for a single user.

        Iterates over each position in the recommendation list, compares the
        item against the corresponding baseline item, and prints the
        miscalibration change caused by each swap.

        Parameters
        ----------
        user_id : str
            User identifier whose history is to be explained.
        """
        cut_value = self.df_2["ORDER"].max()

        base_list = self.df_3[self.df_3["USER_ID"] == int(user_id)].iloc[:cut_value].copy()
        rec_list = self.df_2[self.df_2["USER_ID"] == int(user_id)]

        for base_item, rec_item in zip(base_list.itertuples(), rec_list.itertuples()):
            print("-" * 100)
            p = str(getattr(rec_item, "ORDER"))
            print("* * Baseline: ", base_item)
            print("* * Calibrated:", rec_item)

            if str(getattr(base_item, "ITEM_ID")) == str(getattr(rec_item, "ITEM_ID")):
                print(f"- - The position {p} had no item changed")
                continue

            print(f"- - The position {p} had the item changed: ")
            print(str(getattr(base_item, "ITEM_ID")), " - ", str(getattr(rec_item, "ITEM_ID")))

            base_dist = self.compute_distribution(base_list.head(int(p)).copy())
            calib_base = self.compute_miscalibration(
                self.target_dist[str(user_id)],
                base_dist[str(user_id)]
            )

            realized_dist = self.compute_distribution(rec_list.head(int(p)).copy())
            calib_rec = self.compute_miscalibration(
                self.target_dist[str(user_id)],
                realized_dist[str(user_id)]
            )

            print(f"- - The miscalibration goes from {calib_base} To {calib_rec}")
