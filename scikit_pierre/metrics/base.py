from numpy import mean
from pandas import DataFrame

from ..distributions.accessible import distributions_funcs
from ..distributions.compute_distribution import computer_users_distribution_dict, \
    transform_to_vec
from ..measures.accessible import calibration_measures_funcs
from ..models.item import ItemsInMemory


class BaseMetric:
    """
    Abstract base class for all evaluation metrics.

    Provides shared infrastructure for user-level grouping, ordering, and
    the map-reduce computation pattern used by accuracy, diversity, and
    unexpectedness metrics.

    The semantic meaning of ``df_1``, ``df_2``, and ``df_3`` depends on the
    concrete subclass:

    - ``df_1`` — user interaction profile, test-item set, or baseline list.
    - ``df_2`` — recommendation list being evaluated.
    - ``df_3`` — candidate items, baseline recommendations, or a second
      comparison set (optional).
    """

    def __init__(
            self,
            df_1: DataFrame = None, df_2: DataFrame = None, df_3: DataFrame = None
    ):
        """
        Parameters
        ----------
        df_1 : DataFrame, optional
            User profiles or test-item set.
        df_2 : DataFrame, optional
            Recommendation lists to evaluate.
        df_3 : DataFrame, optional
            Candidate items, baseline list, or comparison set.
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


class BaseCalibrationMetric(BaseMetric):
    """
    Abstract base class for calibration-based evaluation metrics.

    Extends :class:`BaseMetric` with distribution computation helpers,
    a configurable divergence measure, and lazy target/realized distribution
    caching.
    """

    def __init__(
            self,
            users_profile_df: DataFrame, users_rec_list_df: DataFrame, items_set_df: DataFrame,
            distribution_name: str = "CWS", distance_func_name: str = "KL",
            target_dist: dict = None, realized_dist: dict = None
    ):
        """
        Parameters
        ----------
        users_profile_df : DataFrame
            User interaction history used to compute the target distribution.
        users_rec_list_df : DataFrame
            Recommendation lists used to compute the realized distribution.
        items_set_df : DataFrame
            Item catalogue with ``ITEM_ID`` and ``GENRES`` columns.
        distribution_name : str, optional
            Acronym for the genre distribution strategy.  Defaults to
            ``"CWS"``.
        distance_func_name : str, optional
            Acronym for the calibration divergence measure.  Defaults to
            ``"KL"`` (Kullback-Leibler divergence).
        target_dist : dict, optional
            Pre-computed target distributions keyed by user ID.  When
            provided, the distribution computation step is skipped.
        realized_dist : dict, optional
            Pre-computed realized distributions keyed by user ID.
        """
        super().__init__(df_1=users_profile_df, df_2=users_rec_list_df)
        self.target_dist = target_dist
        self.realized_dist = realized_dist

        self.items_df = items_set_df
        self._item_in_memory = None

        self.dist_func = distributions_funcs(distribution=distribution_name)
        self.dist_name = distribution_name

        self.calib_measure_func = calibration_measures_funcs(measure=distance_func_name)
        self.calib_measure_name = distance_func_name

        self.users_ix = None

    def item_preparation(self) -> None:
        """
        Initialise the in-memory item catalogue from ``self.items_df``.

        Loads items by genre into ``self._item_in_memory`` so that
        distribution functions can access per-item class information.
        """
        self._item_in_memory = ItemsInMemory(data=self.items_df)
        self._item_in_memory.item_by_genre()

    @staticmethod
    def transform_to_vec(target_dist: dict, realized_dist: dict):
        """
        Align two genre-keyed distribution dicts into parallel numeric vectors.

        Thin wrapper around
        :func:`~scikit_pierre.distributions.compute_distribution.transform_to_vec`.

        Parameters
        ----------
        target_dist : dict
            Target distribution mapping ``genre -> float``.
        realized_dist : dict
            Realized distribution mapping ``genre -> float``.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(p, q)`` aligned over the union of genre keys.
        """
        p, q = transform_to_vec(target_dist, realized_dist)
        return p, q

    def compute_distribution(self, set_df: DataFrame) -> dict:
        """
        Compute per-user genre distributions from an interaction DataFrame.

        Parameters
        ----------
        set_df : DataFrame
            User interactions or recommendation slice with columns
            ``USER_ID``, ``ITEM_ID``, and a value column.

        Returns
        -------
        dict
            Mapping of ``user_id -> {genre: value}`` for all users in
            *set_df*.
        """
        dist_dict = computer_users_distribution_dict(
            interactions_df=set_df, items_df=self.items_df,
            distribution=self.dist_name
        )
        return dist_dict

    def compute_target_dist(self):
        if self.target_dist is None:
            self.target_dist = self.compute_distribution(self.df_1)

    def compute_realized_dist(self):
        if self.realized_dist is None:
            self.realized_dist = self.compute_distribution(self.df_2)

    def compute(self):
        """

        :return:
        """
        self.checking_users()
        self.compute_target_dist()
