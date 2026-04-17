"""
Genre-to-probability transformation utilities.

Provides functions that convert a raw item catalogue (with pipe-separated
genre strings) into a probability-encoded DataFrame suitable for
distribution-based calibration algorithms.
"""
from pandas import DataFrame, concat


def genre_probability_approach(item_set: DataFrame) -> DataFrame:
    """
    Convert a pipe-separated genre catalogue into a per-item genre probability matrix.

    Each item's genres receive equal weight: if an item belongs to *k* genres
    each genre is assigned a probability of 1/k.  Genres absent from an item
    receive 0.0 (filled via ``fillna``).

    Parameters
    ----------
    item_set : DataFrame
        Must contain the columns ``ITEM_ID`` (item identifier) and ``GENRES``
        (pipe-separated genre string, e.g. ``"Action|Drama"``).

    Returns
    -------
    DataFrame
        Index is ``ITEM_ID``, columns are unique genre names, and cell values
        are the fractional genre probabilities for each item.
    """
    item_classes = []
    for row in item_set.itertuples():
        item_id = getattr(row, "ITEM_ID")
        item_genre = getattr(row, "GENRES")

        splitted = item_genre.split('|')
        genre_ratio = 1.0 / len(splitted)
        genres = {genre: genre_ratio for genre in splitted}
        item_classes.append(DataFrame.from_records(genres, index=[item_id]))
    items_classes_set = concat(item_classes).fillna(0.0)
    return items_classes_set
