"""
Abstract base class for calibration trade-off algorithms.

Defines the shared initialisation logic (data validation, item loading,
batch configuration) and the ``fit`` entry-point that all concrete
trade-off classes must call via ``super()``.
"""
from copy import deepcopy
from pandas import DataFrame

from ..models.item import ItemsInMemory


class BaseTradeOff:
    """
    Abstract base class for calibration trade-off re-ranking algorithms.

    Validates input DataFrames, builds the in-memory item catalogue, and
    stores all shared state (environment config, batch size).  Concrete
    subclasses must call :meth:`env` to register their configuration dict
    before invoking :meth:`fit`.
    """

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame,
                 item_set: DataFrame, users_distribution: DataFrame = None, batch: int = 128):
        """
        Parameters
        ----------
        users_preferences : DataFrame
            User interaction history.  Must contain
            ``USER_ID``, ``ITEM_ID``, and ``TRANSACTION_VALUE``.
        candidate_items : DataFrame
            Pre-ranked candidate items per user.  Must contain
            ``USER_ID``, ``ITEM_ID``, and ``TRANSACTION_VALUE``
            (or ``PREDICTED_VALUE`` from a recommender model).
        item_set : DataFrame
            Item catalogue.  Must contain at least ``ITEM_ID`` and
            ``GENRES`` (pipe-separated genre string).
        users_distribution : DataFrame, optional
            Pre-computed target distributions indexed by ``USER_ID``.
            When provided, the distribution step is skipped and these
            values are used as-is.
        batch : int, optional
            Number of users processed per batch during :meth:`fit`.
            Defaults to 128.

        Raises
        ------
        KeyError
            If the required columns are absent from both *users_preferences*
            and *candidate_items*.
        NameError
            If any item ID in the interaction data is missing from
            *item_set*.
        """
        # At least one of the two DataFrames must have the required columns.
        if {'USER_ID', 'ITEM_ID', 'TRANSACTION_VALUE'}.issubset(set(users_preferences.columns)) or \
                {'USER_ID', 'ITEM_ID', 'TRANSACTION_VALUE'}.issubset(set(candidate_items.columns)):
            self.users_preferences = deepcopy(users_preferences)
            self.candidate_items = deepcopy(candidate_items)
        else:
            raise KeyError("Some column is missing.")

        set_1 = set({str(ix) for ix in users_preferences['ITEM_ID'].unique().tolist() +
                     candidate_items['ITEM_ID'].unique().tolist()})
        set_2 = set({str(ix) for ix in item_set['ITEM_ID'].unique().tolist()})

        if len(set_1 - set_2) > 0:
            print(set_1 - set_2)
            raise NameError("Some wrong information in the ITEM ID.")

        self.item_set = deepcopy(item_set)

        self._item_in_memory = ItemsInMemory(data=self.item_set)
        self.users_distribution = users_distribution
        if self.users_distribution is not None:
            self.users_distribution.fillna(0, inplace=True)

        self.environment = {}
        self.batch = batch

    def env(self, environment: dict) -> None:
        """
        Store the experiment configuration dictionary.

        Parameters
        ----------
        environment : dict
            Key-value pairs defining the algorithm configuration (e.g.
            distribution name, fairness measure, list size, alpha).
            Subclasses define the exact expected keys.
        """
        self.environment = environment

    def fit(self) -> None:
        """
        Guard that the environment has been configured before computation begins.

        Raises
        ------
        SystemError
            If :meth:`env` has not been called (``self.environment`` is
            empty).
        """
        if not self.environment:
            raise SystemError("The configuration need to be set!")
