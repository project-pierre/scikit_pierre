from pandas import DataFrame, concat


def genre_probability_approach(item_set: DataFrame) -> DataFrame:
    """
    Function to deal with the class approach

    :param item_set: A DataFrame with the set of items with the columns: ['ITEM_ID', 'CLASSES']

    :return: A Dataframe were the lines are the items, the columns are the genres and the cells are probability values.
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
