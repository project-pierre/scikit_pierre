"""
Item domain model and in-memory item catalogue.

Defines ``Item``, a lightweight data-carrier for a single catalogue item,
and ``ItemsInMemory``, which loads a DataFrame of items, builds an in-memory
lookup dictionary, and exposes helpers used by the distribution and
calibration modules.
"""
import itertools
from collections import Counter

from copy import deepcopy

from pandas import DataFrame, merge, concat


class Item:
    """
    Data carrier for a single catalogue item.

    Attributes
    ----------
    id : any
        Unique item identifier (usually an integer or string).
    score : float or None
        Predicted or observed interaction value for a specific user context.
        ``None`` until assigned by a data-loading helper.
    classes : dict
        Mapping of class/genre label to fractional weight.  Weights
        typically sum to 1.0 for probability-property distributions.
    bias : float or None
        Precomputed item bias term used by ``LogarithmBias`` calibration.
        ``None`` unless loaded via :meth:`ItemsInMemory.item_by_bias`.
    time : float or None
        Normalised timestamp or reciprocal-rank position weight.
        ``None`` until assigned by a data-loading helper.
    position : int or None
        Rank position in a recommendation list, set during re-ranking.
    """

    def __init__(self, _id, classes: dict, score: float = None, bias: float = None,
                 time: float = None):
        """
        Parameters
        ----------
        _id : any
            Unique item identifier.
        classes : dict
            Mapping of genre/class label to fractional weight.
        score : float, optional
            Predicted or observed interaction value.
        bias : float, optional
            Precomputed item bias term.
        time : float, optional
            Normalised temporal weight or reciprocal-rank position.
        """
        self.id = _id
        self.score = score
        self.classes = classes
        self.bias = bias
        self.time = time
        self.position = None

    def get_id(self):
        """
        Return the item's unique identifier.

        Returns
        -------
        any
            The value of ``self.id``.
        """
        return self.id

    def get_score(self):
        """
        Return the item's score in the current user context.

        Returns
        -------
        float or None
            The value of ``self.score``; ``None`` if not yet assigned.
        """
        return self.score


class ItemsInMemory:
    """
    In-memory catalogue of :class:`Item` objects.

    Loads a raw items DataFrame and exposes several loading strategies
    (by genre, by popularity, by bias) that populate ``self.items``.
    Also provides one-hot encoding and per-user item selection helpers
    consumed by the distribution and calibration modules.

    Attributes
    ----------
    items : dict
        Mapping of str(item_id) -> :class:`Item` instance, populated by
        one of the ``item_by_*`` loading methods.
    encoded : DataFrame or None
        One-hot encoded item feature matrix produced by
        :meth:`one_hot_encode`.
    uniques_genres : list or None
        Sorted list of unique genre labels used as columns in
        :attr:`encoded`.
    """

    def __init__(self, data: DataFrame):
        """
        Parameters
        ----------
        data : DataFrame
            Must contain at least the columns ``ITEM_ID`` and ``GENRES``
            (pipe-separated genre string).
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
        Populate ``self.items`` using uniform genre weights.

        Each item's genres receive equal weight (1 / number_of_genres).
        Populates ``self.items`` as a side effect; returns nothing.
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
        Populate ``self.items`` using popularity group labels with weight 1.0.

        Expects the ``GENRES`` column to contain pre-computed popularity group
        labels (e.g. ``"G01|G02"``).  Each group is assigned a weight of 1.0
        rather than a fractional split.  Exits the process with a descriptive
        message if the ``GENRES`` column is missing.

        Raises
        ------
        SystemExit
            If ``GENRES`` is not a column of the underlying DataFrame.
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
        Classify items into popularity decile groups and populate ``self.items``.

        Counts how many times each item appears in ``users_transactions``,
        normalises by the maximum count, and assigns a decile label
        (``"G01"`` … ``"G10"``).  Items with a normalised popularity ratio
        close to 1.0 receive ``"G01"`` (most popular); those close to 0.0
        receive ``"G10"`` (least popular).

        Parameters
        ----------
        users_transactions : DataFrame
            Must contain at least the column ``ITEM_ID``.
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
        Return the one-hot encoded item feature matrix.

        Returns
        -------
        DataFrame or None
            The DataFrame produced by :meth:`one_hot_encode`, or ``None`` if
            that method has not yet been called.
        """
        return self.encoded

    @staticmethod
    def _getting_genres(row):
        """
        Extract the item identifier and split genre list from a DataFrame row.

        Parameters
        ----------
        row : pandas named-tuple
            A row produced by ``DataFrame.itertuples()``.  Must expose the
            attributes ``ITEM_ID`` and ``GENRES``.

        Returns
        -------
        tuple[any, list[str]]
            ``(item_id, [genre1, genre2, ...])``
        """
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        return item_id, splitted

    def _map_encode(self, row):
        """
        Produce a binary genre presence vector for a single item row.

        Requires ``self.uniques_genres`` to be set (i.e. :meth:`one_hot_encode`
        must be called first).

        Parameters
        ----------
        row : pandas named-tuple
            A row produced by ``DataFrame.itertuples()``.

        Returns
        -------
        tuple[any, list[int]]
            ``(item_id, [1_or_0, ...])`` aligned with ``self.uniques_genres``.
        """
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        return item_id, [1 if genre in splitted else 0 for genre in self.uniques_genres]

    def one_hot_encode(self):
        """
        Build a one-hot genre encoding for all items in ``self._data``.

        Collects all unique genres across the catalogue, then creates a
        binary matrix stored in ``self.encoded`` (index = item IDs,
        columns = unique genre labels).  Also sets ``self.uniques_genres``.
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
        Build a user-specific item dictionary from a user interaction slice.

        Copies each referenced :class:`Item` from ``self.items``, then
        attaches the interaction score and, if available, a normalised
        timestamp or reciprocal-rank position weight.

        Parameters
        ----------
        data : DataFrame
            A per-user interaction slice.  Expected columns:

            - ``ITEM_ID`` (required)
            - ``TRANSACTION_VALUE`` *or* ``PREDICTED_VALUE`` (at least one)
            - ``TIMESTAMP`` (optional) — normalised to ``[0, 1]`` using the
              min/max within the slice; used as ``item.time``.
            - ``ORDER`` (optional) — position index; sets
              ``item.time = 1 / ORDER``.

        Returns
        -------
        dict
            Mapping of str(item_id) -> deep-copied :class:`Item` with
            ``score`` and (when applicable) ``time`` populated.
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
        Convert a recommendation-list item dictionary to a DataFrame.

        Parameters
        ----------
        items : dict
            Mapping of item_id -> :class:`Item`.  Each ``Item`` must have
            ``id`` and ``position`` set.

        Returns
        -------
        DataFrame
            Two-column DataFrame with ``ITEM_ID`` and ``ORDER`` (position).
        """
        user_results = []
        for _, item in items.items():
            user_results += [DataFrame(data=[[item.id, item.position]],
                                       columns=["ITEM_ID", "ORDER"])]
        return concat(user_results, sort=False)

    def transform_to_pandas_items(self) -> DataFrame:
        """
        Convert ``self.items`` back into the original two-column DataFrame format.

        Reconstructs the ``GENRES`` string by joining the keys of each item's
        ``classes`` dict with ``"|"``.

        Returns
        -------
        DataFrame
            Two-column DataFrame with ``ITEM_ID`` and ``GENRES``
            (pipe-separated genre string).
        """
        user_results = []
        for _, item in self.items.items():
            genres = "|".join([g for g in item.classes.keys()])
            user_results += [DataFrame(data=[[item.id, genres]],
                                       columns=["ITEM_ID", "GENRES"])]
        return concat(user_results, sort=False)
