# ############################################################################################### #
# ######################################### Time Based ########################################## #
# ############################################################################################### #
import math
from collections import defaultdict

from numpy import mean


def time_weighted_based(items: dict) -> dict:
    """
    The Time Weight Based - (TWB). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute():
        for index, item in items.items():
            for genre, genre_value in item.classes.items():
                numerator[genre] = numerator.get(genre, 0) + item.time * item.score * genre_value
                denominator[genre] = denominator.get(genre, 0) + item.score

    def genre(g):
        if (g in denominator.keys() and denominator[g] > 0.0) and (g in numerator.keys() and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        else:
            return 0.00001

    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def temporal_slide_window(items: dict, major: int = 10) -> dict:
    """
    The Temporal Slide Window - (TSW). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param major: TODO

    :return: A Dict of genre and value.
    """
    minor = major * 2
    total = len(items)
    batch_size = math.floor(total / minor)
    floor = 0
    top = 0
    distribution_list = []
    enum_items = list(enumerate(list(items.items())))
    for i in range(2, minor + 1):
        top = i * batch_size
        if i == minor:
            to_use_items = dict({item for index, item in enum_items[floor:]})
        else:
            to_use_items = dict(item for index, item in enum_items[floor:top])
        distribution_list.append(time_weighted_based(to_use_items))
        floor = (i - 1) * batch_size

    distribution_list.append(time_weighted_based({
        **dict(item for index, item in enum_items[:batch_size]), **dict(item for index, item in enum_items[floor:])
    }))

    dd = defaultdict(list)
    for d in distribution_list:  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)

    distribution = {}
    for key, value in dd.items():
        distribution[key] = mean(value)

    return distribution


# ############################################################################################### #
# ######################################### Unrevised ########################################## #
# ############################################################################################### #
def time_weighted_probability_genre(items: dict) -> dict:
    """
    The Time Weighted Probability Genre Distribution - (TWPGD). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    dist = time_weighted_based(items)
    norm = sum(dist.values())
    distribution = {g: dist[g] / norm for g in dist}
    return distribution


def time_genre(items: dict) -> dict:
    """
    The Time Genre Distribution - (TGD). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute():
        for index, item in items.items():
            for genre, genre_value in item.genres.items():
                numerator[genre] = numerator.get(genre, 0.) + item.time * genre_value
                denominator[genre] = denominator.get(genre, 0.) + item.time

    compute()
    distribution = {g: numerator[g] / denominator[g] for g in numerator}
    return distribution


def time_probability_genre(items: dict) -> dict:
    """
    The Time Probability Genre Distribution - (TPGD). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    dist = time_genre(items)
    norm = sum(dist.values())
    distribution = {g: dist[g] / norm for g in dist}
    return distribution
