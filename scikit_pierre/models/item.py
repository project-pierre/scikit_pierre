"""
This file contains the Item class used to storage the item attributes.
"""
import itertools
from collections import Counter

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

    def get_id(self):
        """
        Basic get id method
        :return:
        """
        return self.id

    def get_score(self):
        """
        Basic get score method
        :return:
        """
        return self.score


class ItemsInMemory:
    """
    Main class to deal with the items in memory.
    """
    def __init__(self, data: DataFrame):
        """
        Init method
        :param data: this dataframe has two columns [ITEM_ID, GENRES]
        """
        self.items = {}
        self._data = data
        self.encoded = None
        self.uniques_genres = None

    @staticmethod
    def _map_by_genre(row):
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        genre_ratio = 1.0 / len(splitted)

        return Item(_id=item_id, classes={genre: genre_ratio for genre in splitted})

    def item_by_genre(self):
        """
        Method to translate the Dataframe to an Item class
        :return:
        """
        for row in self._data.itertuples():
            item_id = str(getattr(row, "ITEM_ID"))
            item_genre = getattr(row, "GENRES")

            splitted = item_genre.split('|')
            genre_ratio = 1.0 / len(splitted)

            item = Item(_id=item_id, classes={genre: genre_ratio for genre in splitted})
            self.items[item_id] = item

    def item_by_popularity(self):
        """
        Method to translate the Dataframe to an Item class
        :return:
        """
        for row in self._data.itertuples():
            item_id = str(getattr(row, "ITEM_ID"))
            try:
                item_genre = getattr(row, "GENRES")
            except AttributeError:
                print(
                    "We do not find the column called GENRES "
                    "with the popularity groups of each item.\n"
                    "Please verify the example directory to understand how to produce the groups."
                )
                exit(0)

            splitted = item_genre.split('|')
            genre_ratio = 1.0

            item = Item(_id=item_id, classes={genre: genre_ratio for genre in splitted})
            self.items[item_id] = item

    def classifying_item_by_popularity(self, users_transactions: DataFrame):
        """
        Method to translate the Dataframe to an Item class
        :return:
        """
        def group_by_ratio(value: float) -> str:
            if 0.0 <= value < 0.1:
                return 'G10'
            elif 0.1 <= value < 0.2:
                return 'G09'
            elif 0.2 <= value < 0.3:
                return 'G08'
            elif 0.3 <= value < 0.4:
                return 'G07'
            elif 0.4 <= value < 0.5:
                return 'G06'
            elif 0.5 <= value < 0.6:
                return 'G05'
            elif 0.6 <= value < 0.7:
                return 'G04'
            elif 0.7 <= value < 0.8:
                return 'G03'
            elif 0.8 <= value < 0.9:
                return 'G02'
            elif 0.9 <= value <= 1:
                return 'G01'
            return 'G00'

        dict_of_items = Counter(users_transactions["ITEM_ID"].tolist())
        max_value = max(dict_of_items.values())
        count_items_trans = dict(sorted(
            dict_of_items.items(), key=lambda i: i[1], reverse=True
        ))

        for ix, vl in count_items_trans.items():
            self.items[ix] = Item(_id=ix, classes={group_by_ratio(vl/max_value): 1})

    def get_encoded(self) -> DataFrame:
        """
        Method to get encoded data
        :return:
        """
        return self.encoded

    @staticmethod
    def _getting_genres(row):
        """
        Method to get genres data
        :param row:
        :return:
        """
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        return item_id, splitted

    def _map_encode(self, row):
        """

        :param row:
        :return:
        """
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        return item_id, [1 if genre in splitted else 0 for genre in self.uniques_genres]

    def one_hot_encode(self):
        """
        Method to encode encoded data
        :return:
        """
        self.uniques_genres = list(set(itertools.chain.from_iterable([
            genres.split("|")
            for genres in self._data["GENRES"].tolist()
        ])))

        list_of_items = list(map(self._map_encode, self._data.itertuples()))

        self.encoded = DataFrame(
            data=[t[1] for t in list_of_items],
            columns=self.uniques_genres, index=[t[0] for t in list_of_items]
        )
        self.encoded.fillna(0)

    def item_by_bias(self, bias_data: DataFrame):
        """
        Create a dictionary of item id to Item lookup.
        """
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
            self.items[str(item_id)] = item

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
            item_id = str(getattr(row, "ITEM_ID"))
            item = self.items[item_id]
            user_items[item_id] = deepcopy(item)
            user_items[item_id].score = getattr(row, feedback_column)
            if 'TIMESTAMP' in data.columns.tolist():
                upper = getattr(row, 'TIMESTAMP') - minimum
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
        """
        Method to transform the class items in a pandas Dataframe.
        :param items:
        :return:
        """
        user_results = []
        for _, item in items.items():
            user_results += [DataFrame(data=[[item.id, item.position]],
                                       columns=["ITEM_ID", "ORDER"])]
        return concat(user_results, sort=False)

    def transform_to_pandas_items(self) -> DataFrame:
        """
        Method to transform the class items in a pandas Dataframe.
        :param items:
        :return:
        """
        user_results = []
        for _, item in self.items.items():
            genres = "|".join([g for g in item.classes.keys()])
            user_results += [DataFrame(data=[[item.id, genres]],
                                       columns=["ITEM_ID", "GENRES"])]
        return concat(user_results, sort=False)
