"""
This file contains the Item class used to storage the item attributes.
"""

from copy import deepcopy

from pandas import DataFrame, merge, concat


class Item:
    """
    The Item model to be used by the system.
    """

    def __init__(self, _id, classes: dict, score: float = None, bias: float = None,
                 time: float = None):
        """
        :param _id:
        :param classes:
        :param score:
        :return:
        """
        self.id = _id
        self.score = score
        self.classes = classes
        self.bias = bias
        self.time = time
        self.position = None


class ItemsInMemory:

    def __init__(self, data: DataFrame):
        self.items = {}
        self._data = data

    def item_by_genre(self):
        for row in self._data.itertuples():
            item_id = getattr(row, "ITEM_ID")
            item_genre = getattr(row, "GENRES")

            splitted = item_genre.split('|')
            genre_ratio = 1.0 / len(splitted)

            item = Item(_id=item_id, classes={genre: genre_ratio for genre in splitted})
            self.items[item_id] = item

    def item_by_bias(self, bias_data: DataFrame):
        """Create a dictionary of item id to Item lookup."""
        item_bias_data = merge(
            self._data, bias_data,
            how='left', left_on='ITEM_ID', right_on='ITEM_ID'
        )
        self._data = item_bias_data
        for row in self._data.itertuples():
            item_id = getattr(row, 'ITEM_ID')
            item_genre = getattr(row, 'GENRES')
            item_bias = getattr(row, 'BIAS_VALUE')

            splitted = item_genre.split('|')
            genre_ratio = 1. / len(splitted)
            item_genre = {genre: genre_ratio for genre in splitted}

            item = Item(_id=item_id, classes=item_genre, bias=item_bias)
            self.items[item_id] = item

    def select_user_items(self, data: DataFrame) -> dict:
        """
        Function to select items used in the DataFrame.
        :param data: A Pandas DataFrame with three or four columns:
            [USER_ID, ITEM_ID, TRANSACTION_VALUE, TIMESTAMP] or [USER_ID, ITEM_ID, PREDICTED_VALUE].

        :return: A subset of variable items.
        """
        user_items = {}
        feedback_column = "PREDICTED_VALUE"
        maximum = 0
        minimum = 0
        if "TRANSACTION_VALUE" in data.columns.tolist():
            feedback_column = "TRANSACTION_VALUE"
        if 'TIMESTAMP' in data.columns.tolist():
            maximum = data['TIMESTAMP'].max()
            minimum = data['TIMESTAMP'].min()

        for row in data.itertuples():
            item_id = getattr(row, "ITEM_ID")
            item = self.items[item_id]
            user_items[item_id] = deepcopy(item)
            user_items[item_id].score = getattr(row, feedback_column)
            if 'TIMESTAMP' in data.columns.tolist():
                upper = (getattr(row, 'TIMESTAMP') - minimum)
                divisor = maximum - minimum
                try:
                    user_items[item_id].time = upper / divisor
                except ZeroDivisionError:
                    if divisor == 0 and upper == 0:
                        user_items[item_id].time = 1
                    elif divisor == 0:
                        user_items[item_id].time = upper / 1
                    else:
                        user_items[item_id].time = 1 / divisor

            if 'ORDER' in data.columns.tolist():
                user_items[item_id].time = float(1 / int(getattr(row, "ORDER")))
        return user_items

    @staticmethod
    def transform_to_pandas(items: dict) -> DataFrame:
        user_results = []
        for _, item in items.items():
            user_results += [DataFrame(data=[[item.id, item.position]],
                                       columns=["ITEM_ID", "ORDER"])]
        return concat(user_results, sort=False)
