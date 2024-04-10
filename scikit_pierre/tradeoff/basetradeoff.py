"""
This file contains the Base Class to be inherent by other trade-off implementations.
"""
from copy import deepcopy
from pandas import DataFrame

from ..models.item import ItemsInMemory


class BaseTradeOff:
    """
    Tradeoff superclass. To be used for all Tradeoff classes.
    """

    def __init__(self, users_preferences: DataFrame, candidate_items: DataFrame,
                 item_set: DataFrame, users_distribution: DataFrame = None):
        """
        :param users_preferences: A Pandas Dataframe with three columns
            [USER_ID, ITEM_ID, TRANSACTION_VALUE]
        :param candidate_items: A Pandas Dataframe with three columns
            [USER_ID, ITEM_ID, PREDICTED_VALUE]
        :param item_set: A Pandas Dataframe with at least two columns [ITEM_ID, CLASSES]
        """
        # Verifying if all columns are present in the user model and the candidate items.
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

    def env(self, environment: dict) -> None:
        """
        This method is to config the experiment environment.
        :param environment: TODO: Docstring
        """
        self.environment = environment

    def fit(self) -> None:
        """
        This method is the main method to start the trade-off computation.
        """
        if not self.environment:
            raise SystemError("The configuration need to be set!")
