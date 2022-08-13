from copy import deepcopy
from math import log

from numpy import sign
from pandas import DataFrame, concat, merge

from .basetradeoff import BaseTradeOff
from ..classes.acessible import class_funcs
from ..distributions.accessible import distributions_funcs_pandas, distributions_funcs
from ..measures.accessible import calibration_measures_funcs, SIMILARITY_LIST
from ..relevance.accessible import relevance_measures_funcs
from ..tradeoff_weight.accessible import tradeoff_weights_funcs


class CalibrationBase(BaseTradeOff):

    @staticmethod
    def _tradeoff_sim(lmbda: float, relevance_value: float, fairness_value: float, **kwargs) -> float:
        """
        Tradeoff Balance that considers the similarity measures.

        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param relevance_value: The user relevance value.
        :param fairness_value: The user fairness value.
        :return: The utility value from the tradeoff computation.
        """
        return ((1 - lmbda) * relevance_value) + (lmbda * fairness_value)

    @staticmethod
    def _tradeoff_div(lmbda: float, relevance_value: float, fairness_value: float, **kwargs) -> float:
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
        else:
            return self._tradeoff_div


class LinearCalibration(CalibrationBase):
    """
    The Linear Calibration Tradeoff.

    Implementation based on:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045

    - Steck (2018). https://doi.org/10.1145/3240323.3240372
    """

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame, item_set: DataFrame):
        """
        :param users_preferences: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE].
        :param candidate_items: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, PREDICTED_VALUE].
        :param item_set: A Pandas DataFrame of items.
        """
        # Constructing the instance with the basic
        super().__init__(users_preferences, candidate_items, item_set)
        self._items_distribution = None
        # Creating variables to lead with the equation components as functions
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._tradeoff_balance_component = None
        self._class_approach = None

    def config(self, distribution_component: str = "CWS", class_approach="GENRE_PROBABILITY",
               fairness_component: str = "CHI", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01, d: int = 3):
        """
        Method to config the environment. All variable has default values.

        :param distribution_component: The name of the distribution to be used.
        :param class_approach: The name of the class approach to be used.
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
            "class_approach": class_approach,
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
        self._tradeoff_weight_component = tradeoff_weights_funcs(env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)
        self._class_approach = class_funcs(class_approach=class_approach)

    def fit(self, uuids: list = None):
        """
        Method to create a recommendation list for all users passed by param uuids.

        :param uuids: A list with users unique identification.
        :return: A list of Pandas DataFrame, which each position is a user recommendation list.
        """

        super().fit()

        if self.environment['class_approach'] == "GENRE_PROBABILITY":
            self._item_in_memory.item_by_genre()

        if not uuids:
            uuids = self.users_preferences['USER_ID'].unique()

        recommendation_lists = list(map(self._user_recommendation, uuids))
        return recommendation_lists

    def _user_recommendation(self, uid):
        user_pref = self.users_preferences[self.users_preferences['USER_ID'] == uid]
        user_candidate_items = self.candidate_items[self.candidate_items['USER_ID'] == uid]

        # Target Distribution (p)
        target_dist = self._distribution_component(items=self._item_in_memory.select_user_items(data=user_pref))
        # Tradeoff weight (lambda)
        if self.environment['weight'][:2] == "C@":
            lmbda = self._tradeoff_weight_component
        else:
            lmbda = self._tradeoff_weight_component(dist_vec=list(target_dist.values()))

        # Starting select item algorithm to create the recommendation list
        recommendation_list = self._select_item_component(
            target_distribution=target_dist,
            candidate_items=self._item_in_memory.select_user_items(data=user_candidate_items),
            lmbda=lmbda
        )

        rec_list = merge(self._item_in_memory.transform_to_pandas(items=recommendation_list), user_candidate_items,
                         how="left", on=["ITEM_ID"])

        rec_list["USER_ID"] = uid
        return rec_list

    def _compute_utility(self, target_distribution: dict, temp_rec_items: dict, lmbda: float) -> float:
        """
        The kernel of the Linear Tradeoff Balance.

        :param target_distribution: A Dict with float numbers, which represents the distribution values - p.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param temp_rec_items: A dict with a temporary recommendation list.
        :return: A float between [0;1],
        """
        realized_dist = self._distribution_component(items=temp_rec_items)
        fairness_value = self._fairness_component(p=list(target_distribution.values()),
                                                  q=list(realized_dist.values()),
                                                  d=self.environment["d"])

        relevance_value = self._relevance_component([item.score for _, item in temp_rec_items.items()])

        utility_value = self._tradeoff_balance_component(lmbda=lmbda, relevance_value=relevance_value,
                                                         fairness_value=fairness_value)
        return utility_value

    def _surrogate(self, target_distribution: dict, candidate_items: dict, lmbda: float) -> dict:
        """
        Start with an empty recommendation list,
        loop over the candidate items, during each iteration
        update the list with the item that maximizes the utility function.

        :param target_distribution: A Dict with float numbers, which represents the distribution values - p.
        :param candidate_items: A Dict of Item Class instances, which represents the user candidate items.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :return: A Dict of Item Class instances, which represents the user recommendation list.
        """

        recommendation_list = dict()

        # loop for each position in recommendation list
        for order in range(self.environment['list_size']):
            # start loop variables
            max_utility = -999999999
            best_item = None
            best_id = None
            # loop for test each item in each position
            for i_id, item in candidate_items.items():
                if (i_id not in recommendation_list.keys()) and (i_id is not None):
                    temp_rec_items = deepcopy(recommendation_list)
                    temp_rec_items[i_id] = item

                    utility = self._compute_utility(target_distribution=target_distribution,
                                                    temp_rec_items=temp_rec_items, lmbda=lmbda)
                    if utility > max_utility:
                        max_utility = deepcopy(utility)
                        best_item = deepcopy(item)
                        best_id = deepcopy(i_id)
            if best_id is not None:
                best_item.position = order + 1
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
        else:
            raise Exception("Select item algorithm not found!")


class LogarithmBias(CalibrationBase):
    """
    The Logarithmic Bias Calibration Tradeoff.

    Implementation based on:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112
    """
    BIAS_ALPHA = 0.001
    BIAS_SIGMA = 0.001

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame, item_set: DataFrame):
        """
        :param users_preferences: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, TRANSACTION_VALUE].
        :param candidate_items: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, PREDICTED_VALUE].
        :param item_set: A Pandas DataFrame of items.
        """
        # Constructing the instance with the basic
        super().__init__(users_preferences, candidate_items, item_set)
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
        self._class_approach = None

    def config(self, distribution_component: str = "CWS", class_approach="GENRE_PROBABILITY",
               fairness_component: str = "CHI", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01, d: int = 3):
        """
        Method to config the environment. All variable has default values.

        :param distribution_component: The name of the distribution to be used.
        :param class_approach: The name of the class approach to be used.
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
            "class_approach": class_approach,
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
        self._tradeoff_weight_component = tradeoff_weights_funcs(env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)
        self._class_approach = class_funcs(class_approach=class_approach)

    def fit(self, uuids: list = None):
        """
        Method to create a recommendation list for all users passed by param uuids.

        :param uuids: A list with users unique identification.
        :return: A list of Pandas DataFrame, which each position is a user recommendation list.
        """

        super().fit()

        if not uuids:
            uuids = self.users_preferences['USER_ID'].unique()

        self.transaction_mean = self.users_preferences['TRANSACTION_VALUE'].mean()
        self.item_bias = self._computing_item_bias(self.users_preferences)

        if self.environment['class_approach'] == "GENRE_PROBABILITY":
            self._item_in_memory.item_by_bias(self.item_bias)

        recommendation_lists = list(map(self._user_recommendation, uuids))
        return recommendation_lists

    def _user_recommendation(self, uid):
        user_pref = self.users_preferences[self.users_preferences['USER_ID'] == uid]
        user_candidate_items = self.candidate_items[self.candidate_items['USER_ID'] == uid]

        # Target Distribution (p)
        target_dist = self._distribution_component(items=self._item_in_memory.select_user_items(data=user_pref))
        # Tradeoff weight (lambda)
        if self.environment['weight'][:2] == "C@":
            lmbda = self._tradeoff_weight_component
        else:
            lmbda = self._tradeoff_weight_component(dist_vec=list(target_dist.values()))

        # Starting select item algorithm to create the recommendation list
        recommendation_list = self._select_item_component(
            target_distribution=target_dist,
            candidate_items=self._item_in_memory.select_user_items(data=user_candidate_items),
            lmbda=lmbda
        )

        rec_list = merge(self._item_in_memory.transform_to_pandas(items=recommendation_list), user_candidate_items,
                         how="left", on=["ITEM_ID"])

        rec_list["USER_ID"] = uid
        return rec_list

    def _computing_item_bias(self, users_preferences):
        bias_df = deepcopy(users_preferences)
        items_id_list = bias_df['ITEM_ID'].unique().tolist()
        bias_df['TRANSACTION_VALUE'] -= self.transaction_mean
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
        return sum(user_bias_list) / (LogarithmBias.BIAS_SIGMA + len(user_bias_list)), user_bias_list

    def _compute_utility(self, target_distribution: dict, lmbda: float, temp_rec_items: dict,
                         bias_list: list, i_id: str):
        """
        The kernel of the Linear Tradeoff Balance.
        :param target_distribution: A Dict with float numbers, which represents the distribution values - p.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :param temp_rec_items: A temporary recommendation list.
        :return: A float between [0;1],
        """
        realized_dist = self._distribution_component(items=temp_rec_items)

        fairness_value = self._fairness_component(p=list(target_distribution.values()), q=list(realized_dist.values()),
                                                  d=self.environment["d"])

        relevance_value = self._relevance_component([item.score for _, item in temp_rec_items.items()])

        utility_lin = self._tradeoff_balance_component(lmbda=lmbda, relevance_value=relevance_value,
                                                       fairness_value=fairness_value)
        user_bias, new_bias_list = self._computing_user_bias(user_item_list=temp_rec_items, user_bias_list=bias_list,
                                                             i_id=i_id)
        utility_value = sign(utility_lin) * log(abs(utility_lin) + 1) + user_bias
        return utility_value, new_bias_list

    def _surrogate(self, target_distribution: dict, candidate_items: dict, lmbda: float) -> dict:
        """
        Start with an empty recommendation list,
        loop over the candidate items, during each iteration
        update the list with the item that maximizes the utility function.
        :param target_distribution: A Dict with float numbers, which represents the distribution values - p.
        :param candidate_items: A Dict of Item Class instances, which represents the user candidate items.
        :param lmbda: A float between [0;1], which represent the tradeoff weight.
        :return: A Dict of Item Class instances, which represents the user recommendation list.
        """
        recommendation_dict = dict()
        # loop for each position in recommendation list
        for order in range(self.environment['list_size']):
            # start loop variables
            max_utility = -999999999
            best_item = None
            best_id = None
            best_bias_list = list()
            # loop for test each item in each position
            for i_id, item in candidate_items.items():
                if i_id not in recommendation_dict.keys() and i_id is not None:
                    temp_rec_items = deepcopy(recommendation_dict)
                    temp_rec_items[i_id] = item
                    bias_list = deepcopy(best_bias_list)

                    utility, bias_list = self._compute_utility(
                        lmbda=lmbda, temp_rec_items=temp_rec_items,
                        target_distribution=target_distribution, bias_list=bias_list,
                        i_id=i_id
                    )
                    if utility > max_utility:
                        max_utility = deepcopy(utility)
                        best_item = deepcopy(item)
                        best_id = deepcopy(i_id)
            if best_id is not None:
                best_item.position = order + 1
                recommendation_dict[best_id] = best_item
        return recommendation_dict

    def _select_item_funcs(self, algorithm_name: str):
        """
        Method to choice the algorithm.

        :param algorithm_name: The name of the select item algorithm.
        :return: The choice algorithm function.
        """
        if algorithm_name.upper() == "SURROGATE":
            return self._surrogate
        else:
            raise Exception("Select item algorithm not found!")
