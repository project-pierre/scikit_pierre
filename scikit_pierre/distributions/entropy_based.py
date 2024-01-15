from collections import Counter
from math import log2


def global_local_entropy_based(items: dict) -> dict:
    """
    The Global and Local Entropy Based - (GLEB). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def global_entropy():
        genre_list = [
            genre
            for index, item in items.items()
            for genre, genre_value in item.classes.items()
        ]

        n = len(genre_list)
        total = dict(Counter(genre_list))
        return {t: total[t] / n for t in dict(total)}

    def compute():
        for index, item in items.items():
            for genre, genre_value in item.classes.items():
                ent = -(genre_global[genre] * genre_value)*log2(genre_global[genre] * genre_value)
                numerator[genre] = numerator.get(genre, 0) + item.score * ent
                denominator[genre] = denominator.get(genre, 0) + item.score

    def genre(g):
        if (g in denominator.keys() and denominator[g] > 0.0) and (g in numerator.keys() and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        else:
            return 0.00001

    genre_global = global_entropy()
    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


# ############################################################################################### #
# ######################################### Unrevised ########################################### #
# ############################################################################################### #
def absolute_entropy(items: dict) -> dict:
    """
    The Weighted Genre Distribution - (WGD). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    count = {}
    for index, item in items.items():
        for genre, genre_value in item.genres.items():
            count[genre] = count.get(genre, 0) + 1

    total = sum(count.values())

    distribution = {g: count[g] / total for g in count}

    return distribution
