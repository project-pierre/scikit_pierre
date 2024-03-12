"""
This file contains the distribution functions based on time slide window,
which used the timestamp data.
"""

from collections import defaultdict

import math
from statistics import mean

from .class_based import class_weighted_strategy
from .entropy_based import global_local_entropy_based
from .mixed_based import mixed_gleb_twb
from .time_based import time_weighted_based


def temporal_slide_window_base_function(items: dict, major: int = 10, using: str = "CWS") -> dict:
    """
    The Temporal Slide Window - (TSW). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param major: TODO
    :param using: TODO

    :return: A Dict of genre and value.
    """
    base_distribution = None
    if using == "GLEB":
        base_distribution = global_local_entropy_based
    elif using == "TWB":
        base_distribution = time_weighted_based
    elif using == "GLEB_TWB":
        base_distribution = mixed_gleb_twb
    else:
        base_distribution = class_weighted_strategy
    minor = major * 2
    total = len(items)
    if total > minor:
        batch_size = math.floor(total / minor)
        floor = 0
        distribution_list = []
        enum_items = list(enumerate(list(items.items())))
        for i in range(2, minor + 1):
            top = i * batch_size
            if i == minor:
                to_use_items = dict({item for index, item in enum_items[floor:]})
            else:
                to_use_items = dict(item for index, item in enum_items[floor:top])
            distribution_list.append(base_distribution(to_use_items))
            floor = (i - 1) * batch_size

        distribution_list.append(base_distribution({
            **dict(item for index, item in enum_items[:batch_size]),
            **dict(item for index, item in enum_items[floor:])
        }))

        dd = defaultdict(list)
        for d in distribution_list:  # you can list as many input dicts as you want here
            for key, value in d.items():
                dd[key].append(value)

        distribution = {}
        for key, value in dd.items():
            distribution[key] = mean(value)
    else:
        distribution = base_distribution(items)

    return distribution


def temporal_slide_window_base_function_with_probability_property(items: dict,
                                                                  using: str = "CWS") -> dict:
    """
    The Temporal Slide Window with Probability Property - (TSW_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    distribution = temporal_slide_window_base_function(items=items, using=using)
    total = sum(distribution.values())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


# ############################################################################################### #
# ######################################### Usage ########################################### #
# ############################################################################################### #


def temporal_slide_window(items: dict, using: str = "CWS") -> dict:
    """
    The Temporal Slide Window - (TSW). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def temporal_slide_window_with_probability_property(items: dict, using: str = "CWS") -> dict:
    """
    The Temporal Slide Window with Probability Property - (TSW_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_gleb(items: dict, using: str = "GLEB") -> dict:
    """
    The Temporal Slide Window with Global and Local Entropy Based - (TSW_GLEB).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_gleb_with_probability_property(items: dict, using: str = "GLEB") -> dict:
    """
    The TThe Temporal Slide Window with Global and Local Entropy Based - (TSW_GLEB_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_twb(items: dict, using: str = "TWB") -> dict:
    """
    The Temporal Slide Window with Time Weight Based - (TSW_TWB).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_twb_with_probability_property(items: dict, using: str = "TWB") -> dict:
    """
    The TThe Temporal Slide Window with Global and Local Entropy Based - (TSW_TWB_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_twb_gleb(items: dict, using: str = "GLEB_TWB") -> dict:
    """
    The Temporal Slide Window with Time Weight Based and Global and Local Entropy Based -
    (TSW_TWB_GLEB).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_twb_gleb_with_probability_property(items: dict, using: str = "GLEB_TWB") -> dict:
    """
    The Temporal Slide Window with Time Weight Based and Global and Local Entropy Based -
    (TSW_TWB_GLEB_P).
    The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :param using: TODO

    :return: A Dict of genre and value.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)
