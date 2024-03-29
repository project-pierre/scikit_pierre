from numpy import mean
from pandas import DataFrame

from scikit_pierre.distributions.accessible import distributions_funcs
from scikit_pierre.distributions.compute_distribution import computer_users_distribution_dict
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.models.item import ItemsInMemory


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
                # p.append(0.00001)
                p.append(0.0)

            if column in realized_dist:
                q.append(float(realized_dist[str(column)]))
            else:
                # q.append(0.00001)
                q.append(0.0)

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
