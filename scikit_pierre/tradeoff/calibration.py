"""
This file contains implementations of Calibrated Recommendation Trade-off functions.
"""

import logging
from copy import deepcopy
from math import log, ceil

import numpy as np
from numpy import sign
from pandas import DataFrame, concat, merge
from tqdm import tqdm

from .basetradeoff import BaseTradeOff
from ..distributions.accessible import distributions_funcs
from ..distributions.compute_distribution import transform_to_vec
from ..distributions.compute_tilde_q import compute_tilde_q
from ..measures.accessible import calibration_measures_funcs, SIMILARITY_LIST
from ..relevance.accessible import relevance_measures_funcs
from ..tradeoff_weight.accessible import tradeoff_weights_funcs


class CalibrationBase(BaseTradeOff):
    logger = logging.getLogger(__name__)

    @staticmethod
    def _tradeoff_sim(lmbda: float, relevance_value: float, fairness_value: float,
                      **kwargs) -> float:
        """
        Tradeoff Balance that considers the similarity measures.

        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param relevance_value: The user relevance value.
        :param fairness_value: The user fairness value.
        :return: The utility value from the tradeoff computation.
        """
        return ((1 - lmbda) * relevance_value) + (lmbda * fairness_value)

    @staticmethod
    def _tradeoff_div(lmbda: float, relevance_value: float, fairness_value: float,
                      **kwargs) -> float:
        """
        Tradeoff Balance that considers the distance measures.

        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param relevance_value: The user relevance value.
        :param fairness_value: The user fairness value.
        :return: The utility value from the tradeoff computation.
        """
        return ((1 - lmbda) * relevance_value) - (lmbda * fairness_value)

    def _tradeoff_funcs(self, measure: str):
        """
        Method to choice the tradeoff balance.

        :param measure: A fairness measure name.
        :return: A tradeoff Balance function.
        """
        if measure.upper() in SIMILARITY_LIST:
            return self._tradeoff_sim
        return self._tradeoff_div


class LinearCalibration(CalibrationBase):
    """
    The Linear Calibration Tradeoff.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045

    - Steck (2018). https://doi.org/10.1145/3240323.3240372
    """

    def __init__(
            self,
            users_preferences: DataFrame, candidate_items: DataFrame,
            item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128
    ):
        """
        :param users_preferences: A Pandas DataFrame with four columns
            [USER_ID, ITEM_ID, TRANSACTION_VALUE, TIMESTAMP].
        :param candidate_items: A Pandas DataFrame with three columns
            [USER_ID, ITEM_ID, TRANSACTION_VALUE].
        :param item_set: A Pandas DataFrame of items.
        """
        # Constructing the instance with the basic
        super().__init__(users_preferences, candidate_items, item_set, users_distribution, batch)
        self._items_distribution = None
        # Creating variables to lead with the equation components as functions
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._tradeoff_balance_component = None

    def config(self, distribution_component: str = "CWS",
               fairness_component: str = "CHI", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01,
               d: int = 3):
        """
        Method to config the environment. All variable has default values.

        :param distribution_component: The name of the distribution to be used.
        :param fairness_component: The name of the fairness measure to be used.
        :param relevance_component: The name of the relevance measure to be used.
        :param tradeoff_weight_component: The name of the tradeoff weight to be used.
        :param select_item_component: The name of the select item algorithm to be used.
        :param list_size: The size of the recommendation list.
        :param alpha: Trade-off weight to Realized distribution \tilde{q}
        :param d: Dimension value of Minkowski distance.
        """
        super().env(environment={
            "distribution": distribution_component,
            "fairness": fairness_component,
            "relevance": relevance_component,
            "weight": tradeoff_weight_component,
            "selector": select_item_component,
            "list_size": list_size,
            "alpha": alpha,
            "d": d
        })
        # Load the components as function based on the input.
        self._distribution_component = distributions_funcs(distribution=distribution_component)
        self._fairness_component = calibration_measures_funcs(measure=fairness_component)
        self._relevance_component = relevance_measures_funcs(relevance=relevance_component)
        self._tradeoff_weight_component = tradeoff_weights_funcs(
            env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)

    def fit(self, uuids: list = None) -> DataFrame:
        """
        Method to create a recommendation list for all users passed by param uuids.

        :param uuids: A list with users unique identification.
        :return: A list of Pandas DataFrame, which each position is a user recommendation list.
        """

        super().fit()

        self._item_in_memory.item_by_genre()

        if not uuids:
            uuids = self.users_preferences['USER_ID'].unique().tolist()

        progress = tqdm(total=len(uuids))
        loops = int(ceil(len(uuids) / self.batch))

        recommendation_lists = [
            self._calling_rec(
                uuids=uuids[i * self.batch: (i + 1) * self.batch],
                progress=progress
            )
            for i in range(0, loops)
        ]

        return concat(recommendation_lists)

    def _calling_rec(self, uuids, progress) -> DataFrame:

        recommendation_lists = list(map(self._user_recommendation, uuids))
        progress.update(len(uuids))
        progress.set_description("Calibrating recommendations: ")

        return concat(recommendation_lists)

    def _user_recommendation(self, uid) -> DataFrame:
        user_pref = self.users_preferences[self.users_preferences['USER_ID'] == uid].copy()
        user_candidate_items = self.candidate_items[self.candidate_items['USER_ID'] == uid].copy()

        # Target Distribution (p)

        target_dist = {}
        if self.users_distribution is None:
            target_dist = self._distribution_component(
                items=self._item_in_memory.select_user_items(data=user_pref))
        else:
            target_dist = self.users_distribution.loc[uid].to_dict()
            # target_dist = {
            #     col: value for col, value in zip(self.users_distribution.columns.tolist(),
            #                                      pre_computed_distribution.values.tolist())}

        # Tradeoff weight (lambda)
        if self.environment['weight'][:2] == "C@":
            lmbda = self._tradeoff_weight_component
        elif self.environment['weight'] == "MIT":
            cand_dist = self._distribution_component(
                items=self._item_in_memory.select_user_items(data=user_candidate_items)
            )
            lmbda = self._tradeoff_weight_component(
                dist_vec=user_candidate_items["TRANSACTION_VALUE"].tolist(),
                target_dist=list(target_dist.values()), cand_dist=list(cand_dist.values())
            )
        else:
            lmbda = self._tradeoff_weight_component(dist_vec=list(target_dist.values()))

        # Starting select item algorithm to create the recommendation list
        recommendation_list = self._select_item_component(
            target_distribution=target_dist,
            candidate_items=self._item_in_memory.select_user_items(data=user_candidate_items),
            lmbda=lmbda
        )
        # print({uid: recommendation_list})
        recommendations = self._item_in_memory.transform_to_pandas(items=recommendation_list)
        recommendations.loc[:, "ITEM_ID"] = recommendations["ITEM_ID"].astype(int)
        user_candidate_items.loc[:, "ITEM_ID"] = user_candidate_items["ITEM_ID"].astype(int)

        rec_list = merge(recommendations,
                         user_candidate_items,
                         how="left", on=["ITEM_ID"])

        rec_list["USER_ID"] = uid
        return rec_list

    def _compute_utility(
            self, target_distribution: dict, temp_rec_items: dict, lmbda: float
    ) -> float:
        """
        The kernel of the Linear Tradeoff Balance.

        :param target_distribution: A Dict with float numbers,
            which represents the distribution values - p.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param temp_rec_items: A dict with a temporary recommendation list.
        :return: A float between [0;1],
        """
        realized_dist = self._distribution_component(items=temp_rec_items)
        p, q = transform_to_vec(target_distribution, realized_dist)

        fairness_value = self._fairness_component(
            p=p,
            q=compute_tilde_q(p=p, q=q)
        )

        relevance_value = self._relevance_component(
            [item.score for _, item in temp_rec_items.items()]
        )
        utility_value = self._tradeoff_balance_component(
            lmbda=lmbda, relevance_value=relevance_value, fairness_value=fairness_value)
        return utility_value

    def _surrogate(
            self, target_distribution: dict, candidate_items: dict, lmbda: float
    ) -> dict:
        """
        Start with an empty recommendation list,
        loop over the candidate items, during each iteration
        update the list with the item that maximizes the utility function.

        :param target_distribution: A Dict with float numbers,
            which represents the distribution values - p.
        :param candidate_items: A Dict of Item Class instances,
            which represents the user candidate items.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :return: A Dict of Item Class instances, which represents the user recommendation list.
        """

        recommendation_list = {}

        # loop for each position in recommendation list
        range_list = list(range(1, int(self.environment['list_size']) + 1))

        for order in range_list:
            # start loop variables
            max_utility = -np.inf
            best_item = None
            best_id = None

            # loop for test each item in each position
            for i_id, item in candidate_items.items():
                if (i_id not in recommendation_list.keys()) and (i_id is not None):
                    temp_rec_items = deepcopy(recommendation_list)
                    temp_item = deepcopy(item)
                    temp_item.time = float(1 / int(order))
                    temp_rec_items[i_id] = temp_item

                    utility = self._compute_utility(target_distribution=target_distribution,
                                                    temp_rec_items=temp_rec_items, lmbda=lmbda)

                    if float(utility) > float(max_utility):
                        max_utility = float(deepcopy(utility))
                        best_item = deepcopy(temp_item)
                        best_id = deepcopy(i_id)

            if best_id is not None:
                best_item.position = order
                recommendation_list[best_id] = best_item

        return recommendation_list

    def _select_item_funcs(self, algorithm_name: str = "SURROGATE"):
        """
        Method to choice the algorithm.

        :param algorithm_name: The name of the select item algorithm.
        :return: The choice algorithm function.
        """
        if algorithm_name.upper() == "SURROGATE":
            return self._surrogate
        raise NameError("Select item algorithm not found!")


class LogarithmBias(CalibrationBase):
    """
    The Logarithmic Bias Calibration Tradeoff.

    Implementation based on:

    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112
    """
    BIAS_ALPHA = 0.001
    BIAS_SIGMA = 0.001

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame,
                 item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128):
        """
        :param users_preferences: A Pandas DataFrame with three columns
            [USER_ID, ITEM_ID, TRANSACTION_VALUE].
        :param candidate_items: A Pandas DataFrame with three columns
            [USER_ID, ITEM_ID, PREDICTED_VALUE].
        :param item_set: A Pandas DataFrame of items.
        """
        # Constructing the instance with the basic
        super().__init__(users_preferences, candidate_items, item_set, users_distribution,
                         batch=batch)
        self.item_bias = None
        self.transaction_mean = None
        self._items_distribution = None
        # Creating variables to lead with the equation components as functions
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._tradeoff_balance_component = None

    def config(self, distribution_component: str = "CWS",
               fairness_component: str = "CHI", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01,
               d: int = 3):
        """
        Method to config the environment. All variable has default values.

        :param distribution_component: The name of the distribution to be used.
        :param fairness_component: The name of the fairness measure to be used.
        :param relevance_component: The name of the relevance measure to be used.
        :param tradeoff_weight_component: The name of the tradeoff weight to be used.
        :param select_item_component: The name of the select item algorithm to be used.
        :param list_size: The size of the recommendation list.
        :param alpha: Trade-off weight to Realized distribution \tilde{q}
        :param d: Dimension value of Minkowski distance.
        """
        super().env(environment={
            "distribution": distribution_component,
            "calibration": fairness_component,
            "relevance": relevance_component,
            "weight": tradeoff_weight_component,
            "selector": select_item_component,
            "list_size": list_size,
            "alpha": alpha,
            "d": d
        })
        # Load the components as function based on the input.
        self._distribution_component = distributions_funcs(distribution=distribution_component)
        self._fairness_component = calibration_measures_funcs(measure=fairness_component)
        self._relevance_component = relevance_measures_funcs(relevance=relevance_component)
        self._tradeoff_weight_component = tradeoff_weights_funcs(
            env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)

    def fit(self, uuids: list = None) -> DataFrame:
        """
        Method to create a recommendation list for all users passed by param uuids.

        :param uuids: A list with users unique identification.
        :return: A list of Pandas DataFrame, which each position is a user recommendation list.
        """

        super().fit()

        # self._item_in_memory.item_by_genre()

        if not uuids:
            uuids = self.users_preferences['USER_ID'].unique().tolist()

        self.transaction_mean = self.users_preferences['TRANSACTION_VALUE'].mean()
        self.item_bias = self._computing_item_bias(self.users_preferences)

        self._item_in_memory.item_by_bias(self.item_bias)

        progress = tqdm(total=len(uuids))
        loops = int(ceil(len(uuids) / self.batch))

        recommendation_lists = [
            self._calling_rec(
                uuids=uuids[i * self.batch: (i + 1) * self.batch],
                progress=progress
            )
            for i in range(0, loops)
        ]

        return concat(recommendation_lists)

    def _calling_rec(self, uuids, progress) -> DataFrame:

        recommendation_lists = list(map(self._user_recommendation, uuids))
        progress.update(len(uuids))
        progress.set_description("Calibrating recommendations: ")

        return concat(recommendation_lists)

    def _user_recommendation(self, uid):
        user_pref = self.users_preferences[self.users_preferences['USER_ID'] == uid]
        user_candidate_items = self.candidate_items[self.candidate_items['USER_ID'] == uid]

        # Target Distribution (p)
        target_dist = {}
        if self.users_distribution is None:
            target_dist = self._distribution_component(
                items=self._item_in_memory.select_user_items(data=user_pref))
        else:
            target_dist = self.users_distribution.loc[uid].to_dict()
        # Tradeoff weight (lambda)

        if self.environment['weight'][:2] == "C@":
            lmbda = self._tradeoff_weight_component
        elif self.environment['weight'] == "MIT":
            cand_dist = self._distribution_component(
                items=self._item_in_memory.select_user_items(data=user_candidate_items)
            )
            lmbda = self._tradeoff_weight_component(
                dist_vec=user_candidate_items["TRANSACTION_VALUE"].tolist(),
                target_dist=list(target_dist.values()), cand_dist=list(cand_dist.values())
            )
        else:
            lmbda = self._tradeoff_weight_component(dist_vec=list(target_dist.values()))

        # Starting select item algorithm to create the recommendation list
        recommendation_list = self._select_item_component(
            target_distribution=target_dist,
            candidate_items=self._item_in_memory.select_user_items(data=user_candidate_items),
            lmbda=lmbda
        )
        # print({uid: recommendation_list})
        recommendations = self._item_in_memory.transform_to_pandas(items=recommendation_list)
        recommendations.loc[:, "ITEM_ID"] = recommendations["ITEM_ID"].astype(int)
        user_candidate_items.loc[:, "ITEM_ID"] = user_candidate_items["ITEM_ID"].astype(int)

        rec_list = merge(recommendations,
                         user_candidate_items,
                         how="left", on=["ITEM_ID"])

        rec_list["USER_ID"] = uid
        return rec_list

    def _computing_item_bias(self, users_preferences):
        bias_df = deepcopy(users_preferences)
        items_id_list = bias_df['ITEM_ID'].unique().tolist()
        # bias_df['TRANSACTION_VALUE'] -= self.transaction_mean
        bias_df.loc[:, "TRANSACTION_VALUE"] = bias_df["TRANSACTION_VALUE"].apply(
            lambda x: x - self.transaction_mean
        )

        item_bias_df = DataFrame()
        for item_id in items_id_list:
            item_subset_df = bias_df[bias_df['ITEM_ID'] == item_id]
            numerator = item_subset_df['TRANSACTION_VALUE'].sum()
            denominator = LogarithmBias.BIAS_ALPHA + len(item_subset_df)
            item_bias_df = concat([
                item_bias_df, DataFrame(
                    data=[[item_id, numerator / denominator]],
                    columns=['ITEM_ID', 'BIAS_VALUE']
                )
            ])
        return item_bias_df

    def _computing_user_bias(self, user_item_list, user_bias_list, i_id):
        numerator = user_item_list[i_id].score - self.transaction_mean - user_item_list[i_id].bias
        user_bias_list.append(numerator)
        return sum(user_bias_list) / (
                LogarithmBias.BIAS_SIGMA + len(user_bias_list)), user_bias_list

    def _compute_utility(self, target_distribution: dict, lmbda: float, temp_rec_items: dict,
                         bias_list: list, i_id: str):
        """
        The kernel of the Linear Tradeoff Balance.
        :param target_distribution: A Dict with float numbers,
            which represents the distribution values - p.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param temp_rec_items: A temporary recommendation list.
        :return: A float between [0;1],
        """
        realized_dist = self._distribution_component(items=temp_rec_items)
        p, q = transform_to_vec(target_distribution, realized_dist)

        fairness_value = self._fairness_component(
            p=p,
            q=compute_tilde_q(p=p, q=q)
        )

        relevance_value = self._relevance_component(
            [item.score for _, item in temp_rec_items.items()])

        utility_lin = self._tradeoff_balance_component(lmbda=lmbda, relevance_value=relevance_value,
                                                       fairness_value=fairness_value)
        user_bias, new_bias_list = self._computing_user_bias(user_item_list=temp_rec_items,
                                                             user_bias_list=bias_list,
                                                             i_id=i_id)
        utility_value = sign(utility_lin) * log(abs(utility_lin) + 1) + user_bias
        return utility_value, new_bias_list

    def _surrogate(self, target_distribution: dict, candidate_items: dict, lmbda: float) -> dict:
        """
        Start with an empty recommendation list,
        loop over the candidate items, during each iteration
        update the list with the item that maximizes the utility function.

        :param target_distribution: A Dict with float numbers,
            which represents the distribution values - p.
        :param candidate_items: A Dict of Item Class instances,
            which represents the user candidate items.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :return: A Dict of Item Class instances, which represents the user recommendation list.
        """
        recommendation_list = {}

        range_list = range(1, int(self.environment['list_size']) + 1)

        # loop for each position in recommendation list
        for order in range_list:
            # start loop variables
            max_utility = -np.inf
            best_item = None
            best_id = None
            best_bias_list = []
            # loop for test each item in each position
            for i_id, item in candidate_items.items():
                if (i_id not in recommendation_list.keys()) and (i_id is not None):
                    temp_rec_items = deepcopy(recommendation_list)
                    temp_item = deepcopy(item)
                    bias_list = deepcopy(best_bias_list)
                    temp_item.time = float(1 / int(order))
                    temp_rec_items[i_id] = temp_item

                    utility, bias_list = self._compute_utility(
                        lmbda=lmbda, temp_rec_items=temp_rec_items,
                        target_distribution=target_distribution, bias_list=bias_list,
                        i_id=i_id
                    )
                    if float(utility) > float(max_utility):
                        max_utility = float(deepcopy(utility))
                        best_item = deepcopy(temp_item)
                        best_id = deepcopy(i_id)
                        best_bias_list = bias_list
            if best_id is not None:
                best_item.position = order
                recommendation_list[best_id] = best_item
        return recommendation_list

    def _select_item_funcs(self, algorithm_name: str):
        """
        Method to choice the algorithm.

        :param algorithm_name: The name of the select item algorithm.
        :return: The choice algorithm function.
        """
        if algorithm_name.upper() == "SURROGATE":
            return self._surrogate
        raise NameError("Select item algorithm not found!")


class PopularityCalibration(LinearCalibration):

    def __init__(
            self,
            users_preferences: DataFrame, candidate_items: DataFrame,
            item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128
    ):
        super().__init__(
            users_preferences, candidate_items, item_set, users_distribution, batch
        )


class TwoStageCalibration:

    def __init__(
            self,
            users_preferences: DataFrame, candidate_items: DataFrame,
            item_set: DataFrame, users_popularity_distribution: DataFrame = None,
            users_genres_distribution: DataFrame = None,
            batch: int = 128
    ):
        self.users_preferences = users_preferences
        self.candidate_items = candidate_items
        self.item_set = item_set
        self.users_popularity_distribution = users_popularity_distribution
        self.users_genres_distribution = users_genres_distribution
        self.batch = batch
        if "POPULARITY" not in self.item_set.columns and "GENRES" not in self.item_set.columns:
            raise ("We not found the columns POPULARITY and GENRES in the item_set."
                   "Please insert these columns.")
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._list_size = None
        self._alpha = None
        self._d = None
        self.pop_calib_instance = None
        self.gen_calib_instance = None

    def config(
            self,
            distribution_component: str = "CWS", fairness_component: str = "CHI",
            relevance_component: str = "SUM", tradeoff_weight_component: str = "STD",
            select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01,
            d: int = 3
    ):
        """
        Method to config the environment. All variable has default values.

        :param distribution_component: The name of the distribution to be used.
        :param fairness_component: The name of the fairness measure to be used.
        :param relevance_component: The name of the relevance measure to be used.
        :param tradeoff_weight_component: The name of the tradeoff weight to be used.
        :param select_item_component: The name of the select item algorithm to be used.
        :param list_size: The size of the recommendation list.
        :param alpha: Trade-off weight to Realized distribution \tilde{q}
        :param d: Dimension value of Minkowski distance.
        """
        # Creating variables to lead with the equation components as functions
        self._distribution_component = distribution_component
        self._fairness_component = fairness_component
        self._relevance_component = relevance_component
        self._tradeoff_weight_component = tradeoff_weight_component
        self._select_item_component = select_item_component
        self._list_size = list_size
        self._alpha = alpha
        self._d = d

    def fit(self, uuids: list = None) -> DataFrame:
        """
        Method to create a recommendation list for all users passed by param uuids.

        :param uuids: A list with users unique identification.
        :return: A list of Pandas DataFrame, which each position is a user recommendation list.
        """

        self.pop_calib_instance = PopularityCalibration(
            self.users_preferences, self.candidate_items,
            self.item_set, self.users_popularity_distribution,
        )
        first_list_size = int(ceil(len(self.users_preferences)/len(self.users_preferences["USER_ID"].unique().tolist())/2))
        self.pop_calib_instance.config(
            distribution_component=self._distribution_component,
            fairness_component=self._fairness_component,
            relevance_component=self._relevance_component,
            tradeoff_weight_component=self._tradeoff_weight_component,
            select_item_component=self._select_item_component,
            list_size=first_list_size,
            alpha=self._alpha,
            d=self._d
        )
        popularity_list = self.pop_calib_instance.fit(uuids)

        self.gen_calib_instance = LinearCalibration(
            self.users_preferences, popularity_list,
            self.item_set, self.users_genres_distribution
        )
        self.gen_calib_instance.config(
            distribution_component=self._distribution_component,
            fairness_component=self._fairness_component,
            relevance_component=self._relevance_component,
            tradeoff_weight_component=self._tradeoff_weight_component,
            select_item_component=self._select_item_component,
            list_size=self._list_size,
            alpha=self._alpha,
            d=self._d
        )
        rec_lists = self.gen_calib_instance.fit(uuids)
        return rec_lists