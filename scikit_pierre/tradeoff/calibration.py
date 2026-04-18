"""
Calibrated recommendation re-ranking algorithms.

Provides concrete trade-off implementations that balance relevance and
calibration (fairness) when building a recommendation list from a set of
candidate items.  All classes extend :class:`~basetradeoff.BaseTradeOff`.

Classes
-------
CalibrationBase
    Mixin that adds the relevance/fairness trade-off balance functions.
LinearCalibration
    Standard greedy surrogate optimisation (Silva et al., 2021;
    Steck, 2018).
LogarithmBias
    Extends LinearCalibration with an item-bias correction term
    (Silva et al., 2021).
PopularityCalibration
    Convenience subclass of LinearCalibration for popularity-based
    item sets.
TwoStageCalibration
    Two-stage pipeline: popularity calibration followed by genre
    calibration.
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
    """
    Mixin that adds relevance/fairness trade-off balance helpers.

    Provides two static balance functions — one for similarity measures
    (which are added) and one for divergence measures (which are subtracted)
    — and a router that selects the appropriate one based on the measure name.
    """

    logger = logging.getLogger(__name__)

    @staticmethod
    def _tradeoff_sim(lmbda: float, relevance_value: float, fairness_value: float,
                      **kwargs) -> float:
        """
        Compute the trade-off utility for a similarity-based fairness measure.

        Uses the additive formulation because higher similarity values are
        better:

            utility = (1 - lambda) * relevance + lambda * fairness

        Parameters
        ----------
        lmbda : float
            Trade-off weight in ``[0, 1]``.
        relevance_value : float
            Scalar relevance score for the current candidate list.
        fairness_value : float
            Scalar similarity value between the target and realized
            distributions.
        **kwargs
            Ignored; present for a uniform call signature.

        Returns
        -------
        float
            Utility value for the current candidate list.
        """
        return ((1 - lmbda) * relevance_value) + (lmbda * fairness_value)

    @staticmethod
    def _tradeoff_div(lmbda: float, relevance_value: float, fairness_value: float,
                      **kwargs) -> float:
        """
        Compute the trade-off utility for a divergence-based fairness measure.

        Uses the subtractive formulation because higher divergence values are
        worse (we want to minimise divergence, so it is subtracted):

            utility = (1 - lambda) * relevance - lambda * fairness

        Parameters
        ----------
        lmbda : float
            Trade-off weight in ``[0, 1]``.
        relevance_value : float
            Scalar relevance score for the current candidate list.
        fairness_value : float
            Scalar divergence value between the target and realized
            distributions.
        **kwargs
            Ignored; present for a uniform call signature.

        Returns
        -------
        float
            Utility value for the current candidate list.
        """
        return ((1 - lmbda) * relevance_value) - (lmbda * fairness_value)

    def _tradeoff_funcs(self, measure: str):
        """
        Select the trade-off balance function appropriate for *measure*.

        Parameters
        ----------
        measure : str
            Fairness measure acronym.  Similarity measures (present in
            :data:`~scikit_pierre.measures.accessible.SIMILARITY_LIST`) use
            :meth:`_tradeoff_sim`; all others use :meth:`_tradeoff_div`.

        Returns
        -------
        callable
            Either :meth:`_tradeoff_sim` or :meth:`_tradeoff_div`.
        """
        if measure.upper() in SIMILARITY_LIST:
            return self._tradeoff_sim
        return self._tradeoff_div


class LinearCalibration(CalibrationBase):
    """
    Greedy surrogate calibration trade-off (LinearCalibration).

    Builds a recommendation list position-by-position using a greedy
    surrogate that maximises::

        utility = (1 - lambda) * relevance ± lambda * fairness

    at each step, where *relevance* is the score of the current candidate
    list and *fairness* is the miscalibration (or similarity) between the
    target distribution and the realized distribution of the current list.

    References
    ----------
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
        Parameters
        ----------
        users_preferences : DataFrame
            User interaction history with columns
            ``USER_ID``, ``ITEM_ID``, ``TRANSACTION_VALUE``, and
            optionally ``TIMESTAMP``.
        candidate_items : DataFrame
            Pre-ranked candidate items with columns
            ``USER_ID``, ``ITEM_ID``, ``TRANSACTION_VALUE``
            (or ``PREDICTED_VALUE``).
        item_set : DataFrame
            Item catalogue with at least ``ITEM_ID`` and ``GENRES``.
        users_distribution : DataFrame, optional
            Pre-computed per-user target distributions.
        batch : int, optional
            Users processed per batch.  Defaults to 128.
        """
        super().__init__(users_preferences, candidate_items, item_set, users_distribution, batch)
        self._items_distribution = None
        # Each component is resolved to a callable by config().
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._tradeoff_balance_component = None

    def config(self, distribution_component: str = "CWS",
               fairness_component: str = "CHI_SQUARE", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01,
               d: int = 3):
        """
        Configure the trade-off components.  Must be called before :meth:`fit`.

        Parameters
        ----------
        distribution_component : str, optional
            Acronym for the genre distribution strategy.  Defaults to
            ``"CWS"`` (Class Weighted Strategy).
        fairness_component : str, optional
            Acronym for the calibration/fairness measure.  Defaults to
            ``"CHI_SQUARE"`` (Pearson Chi-Square).
        relevance_component : str, optional
            Acronym for the relevance scoring function.  Defaults to
            ``"SUM"``.
        tradeoff_weight_component : str, optional
            Acronym for the lambda weight function.  Defaults to ``"STD"``
            (normalised standard deviation).
        select_item_component : str, optional
            Acronym for the item selection algorithm.  Defaults to
            ``"SURROGATE"`` (greedy surrogate).
        list_size : int, optional
            Length of the final recommendation list.  Defaults to 10.
        alpha : float, optional
            Smoothing weight applied to the realised distribution
            (tilde-q computation).  Defaults to 0.01.
        d : int, optional
            Order parameter for the Minkowski distance family.
            Defaults to 3.
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
        # Resolve each string acronym to its callable counterpart.
        self._distribution_component = distributions_funcs(distribution=distribution_component)
        self._fairness_component = calibration_measures_funcs(measure=fairness_component)
        self._relevance_component = relevance_measures_funcs(relevance=relevance_component)
        self._tradeoff_weight_component = tradeoff_weights_funcs(
            env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)

    def fit(self, uuids: list = None) -> DataFrame:
        """
        Build calibrated recommendation lists for all (or a subset of) users.

        Parameters
        ----------
        uuids : list, optional
            User IDs to process.  When ``None``, all users present in
            ``users_preferences`` are used.

        Returns
        -------
        DataFrame
            Concatenated recommendation lists for all processed users.
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

        # Compute or retrieve the target distribution p for this user.
        target_dist = {}
        if self.users_distribution is None:
            target_dist = self._distribution_component(
                items=self._item_in_memory.select_user_items(data=user_pref))
        else:
            target_dist = self.users_distribution.loc[uid].to_dict()

        # MIT requires the candidate distribution to compute lambda;
        # C@<value> is a pre-set constant, not a function call.
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

        recommendation_list = self._select_item_component(
            target_distribution=target_dist,
            candidate_items=self._item_in_memory.select_user_items(data=user_candidate_items),
            lmbda=lmbda
        )
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
    Calibration trade-off with an item-bias correction (LogarithmBias).

    Extends the linear calibration by adding a per-item bias term to the
    utility before applying a sign-preserving logarithm, which compresses
    extreme utility values and incorporates item popularity bias estimated
    from the training interactions.

    The item bias is estimated as::

        bias(i) = sum_u(r_ui - mean_r) / (alpha + |ratings(i)|)

    where ``alpha`` (``BIAS_ALPHA``) is a regularisation constant.

    References
    ----------
    - Silva et al. (2021). https://doi.org/10.1016/j.eswa.2021.115112
    """
    BIAS_ALPHA = 0.001
    BIAS_SIGMA = 0.001

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame,
                 item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128):
        """
        Parameters
        ----------
        users_preferences : DataFrame
            User interaction history with columns
            ``USER_ID``, ``ITEM_ID``, ``TRANSACTION_VALUE``.
        candidate_items : DataFrame
            Pre-ranked candidate items with columns
            ``USER_ID``, ``ITEM_ID``, ``PREDICTED_VALUE``.
        item_set : DataFrame
            Item catalogue with at least ``ITEM_ID`` and ``GENRES``.
        users_distribution : DataFrame, optional
            Pre-computed per-user target distributions.
        batch : int, optional
            Users processed per batch.  Defaults to 128.
        """
        super().__init__(users_preferences, candidate_items, item_set, users_distribution,
                         batch=batch)
        self.item_bias = None
        self.transaction_mean = None
        self._items_distribution = None
        # Each component is resolved to a callable by config().
        self._distribution_component = None
        self._fairness_component = None
        self._relevance_component = None
        self._tradeoff_weight_component = None
        self._select_item_component = None
        self._tradeoff_balance_component = None

    def config(self, distribution_component: str = "CWS",
               fairness_component: str = "CHI_SQUARE", relevance_component: str = "SUM",
               tradeoff_weight_component: str = "STD",
               select_item_component: str = "SURROGATE", list_size: int = 10, alpha: float = 0.01,
               d: int = 3):
        """
        Configure the trade-off components.  Must be called before :meth:`fit`.

        Parameters
        ----------
        distribution_component : str, optional
            Acronym for the genre distribution strategy.  Defaults to
            ``"CWS"``.
        fairness_component : str, optional
            Acronym for the calibration measure.  Defaults to ``"CHI_SQUARE"``.
        relevance_component : str, optional
            Acronym for the relevance scoring function.  Defaults to
            ``"SUM"``.
        tradeoff_weight_component : str, optional
            Acronym for the lambda weight function.  Defaults to ``"STD"``.
        select_item_component : str, optional
            Acronym for the item selection algorithm.  Defaults to
            ``"SURROGATE"``.
        list_size : int, optional
            Length of the final recommendation list.  Defaults to 10.
        alpha : float, optional
            Smoothing weight for the tilde-q distribution.  Defaults to
            0.01.
        d : int, optional
            Order for Minkowski distance.  Defaults to 3.
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
        # Resolve each string acronym to its callable counterpart.
        self._distribution_component = distributions_funcs(distribution=distribution_component)
        self._fairness_component = calibration_measures_funcs(measure=fairness_component)
        self._relevance_component = relevance_measures_funcs(relevance=relevance_component)
        self._tradeoff_weight_component = tradeoff_weights_funcs(
            env_lambda=tradeoff_weight_component)
        self._tradeoff_balance_component = self._tradeoff_funcs(measure=fairness_component)
        self._select_item_component = self._select_item_funcs(algorithm_name=select_item_component)

    def fit(self, uuids: list = None) -> DataFrame:
        """
        Build calibrated recommendation lists with bias correction for all users.

        Computes per-item bias from the training interactions before
        delegating to the greedy surrogate loop.

        Parameters
        ----------
        uuids : list, optional
            User IDs to process.  When ``None``, all users present in
            ``users_preferences`` are used.

        Returns
        -------
        DataFrame
            Concatenated recommendation lists for all users.
        """

        super().fit()

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
        recommendations = self._item_in_memory.transform_to_pandas(items=recommendation_list)
        recommendations.loc[:, "ITEM_ID"] = recommendations["ITEM_ID"].astype(int)
        user_candidate_items.loc[:, "ITEM_ID"] = user_candidate_items["ITEM_ID"].astype(int)

        rec_list = merge(recommendations,
                         user_candidate_items,
                         how="left", on=["ITEM_ID"])

        rec_list["USER_ID"] = uid
        return rec_list

    def _computing_item_bias(self, users_preferences: DataFrame) -> DataFrame:
        """
        Estimate per-item bias from user interactions.

        The bias for item *i* is::

            bias(i) = sum_u(r_ui - mean_r) / (BIAS_ALPHA + |ratings(i)|)

        where the mean is taken over all ratings in *users_preferences*.

        Parameters
        ----------
        users_preferences : DataFrame
            Training interactions with ``ITEM_ID`` and
            ``TRANSACTION_VALUE`` columns.

        Returns
        -------
        DataFrame
            Two-column DataFrame with ``ITEM_ID`` and ``BIAS_VALUE``.
        """
        bias_df = deepcopy(users_preferences)
        items_id_list = bias_df['ITEM_ID'].unique().tolist()
        # Centre ratings before summing to make bias sign-interpretable.
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

    def _computing_user_bias(
            self, user_item_list: dict, user_bias_list: list, i_id: str
    ):
        """
        Incrementally update the user bias accumulator with the latest item.

        The user bias is the regularised mean of per-item residuals
        ``(score - global_mean - item_bias)`` over all items added so far.

        Parameters
        ----------
        user_item_list : dict
            Current temporary recommendation dict (item_id -> Item).
        user_bias_list : list of float
            Accumulated residuals from previous positions.
        i_id : str
            Item ID of the most recently added item.

        Returns
        -------
        tuple[float, list[float]]
            ``(user_bias, updated_bias_list)``
        """
        numerator = user_item_list[i_id].score - self.transaction_mean - user_item_list[i_id].bias
        user_bias_list.append(numerator)
        return sum(user_bias_list) / (
                LogarithmBias.BIAS_SIGMA + len(user_bias_list)), user_bias_list

    def _compute_utility(self, target_distribution: dict, lmbda: float, temp_rec_items: dict,
                         bias_list: list, i_id: str):
        """
        Compute the bias-corrected utility for a candidate list state.

        Calculates the linear trade-off utility, applies a sign-preserving
        logarithm to compress extreme values, then adds the running user
        bias to account for item popularity effects.

        Parameters
        ----------
        target_distribution : dict
            Target genre distribution *p* (genre -> float).
        lmbda : float
            Trade-off weight in ``[0, 1]``.
        temp_rec_items : dict
            Temporary recommendation list (item_id -> Item).
        bias_list : list of float
            Accumulated user-bias residuals from previous positions.
        i_id : str
            Item ID of the most recently tentatively added item.

        Returns
        -------
        tuple[float, list[float]]
            ``(utility_value, updated_bias_list)``
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
    """
    Convenience subclass of :class:`LinearCalibration` for popularity-group
    item sets.

    Item sets for this class are expected to use popularity groups (e.g.
    ``"G01"``…``"G10"``) as genre labels instead of content genres.  The
    algorithm is otherwise identical to :class:`LinearCalibration`.
    """

    def __init__(
            self,
            users_preferences: DataFrame, candidate_items: DataFrame,
            item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128
    ):
        super().__init__(
            users_preferences, candidate_items, item_set, users_distribution, batch
        )


class TwoStageCalibration:
    """
    Two-stage calibration pipeline (popularity then genre).

    Stage 1: Runs :class:`PopularityCalibration` on the original candidate
    list to produce an intermediate list of size
    ``ceil(avg_profile_length / 2)``.

    Stage 2: Runs :class:`LinearCalibration` on the intermediate list to
    produce the final genre-calibrated recommendation list.

    This pipeline allows simultaneous calibration over two orthogonal
    dimensions — popularity and genre diversity.
    """

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
        Configure both calibration stages.  Must be called before :meth:`fit`.

        The same configuration is forwarded to both
        :class:`PopularityCalibration` (stage 1) and
        :class:`LinearCalibration` (stage 2).

        Parameters
        ----------
        distribution_component : str, optional
            Acronym for the genre distribution strategy.  Defaults to
            ``"CWS"``.
        fairness_component : str, optional
            Acronym for the calibration measure.  Defaults to ``"CHI"``.
        relevance_component : str, optional
            Acronym for the relevance scoring function.  Defaults to
            ``"SUM"``.
        tradeoff_weight_component : str, optional
            Acronym for the lambda weight function.  Defaults to ``"STD"``.
        select_item_component : str, optional
            Acronym for the item selection algorithm.  Defaults to
            ``"SURROGATE"``.
        list_size : int, optional
            Length of the final (stage 2) recommendation list.  Defaults
            to 10.  Stage 1 list size is derived automatically.
        alpha : float, optional
            Smoothing weight for the tilde-q distribution.  Defaults to
            0.01.
        d : int, optional
            Order for Minkowski distance.  Defaults to 3.
        """
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
        Run the two-stage calibration pipeline and return recommendation lists.

        Stage 1 produces an intermediate list of size
        ``ceil(avg_profile_length / 2)`` via :class:`PopularityCalibration`.
        Stage 2 re-ranks that list using :class:`LinearCalibration` to the
        final ``list_size``.

        Parameters
        ----------
        uuids : list, optional
            User IDs to process.  When ``None``, all users in
            ``users_preferences`` are used.

        Returns
        -------
        DataFrame
            Concatenated final recommendation lists for all users.
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