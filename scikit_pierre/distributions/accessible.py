from . import class_based
from . import weighted_strategy


def distributions_funcs(distribution: str):
    """
    Function to decide what distance measure will be used.

    :param distribution: The acronyms (initials) assigned to a distribution finder, which will be used by.
    :return: The choose function.
    """
    if distribution == "CWS":
        return class_based.class_weighted_strategy
    elif distribution == "WPS":
        return class_based.weighted_probability_strategy
    elif distribution == "TWB":
        return class_based.time_weighted_based
    else:
        raise Exception("Distribution not found!")


def distributions_funcs_pandas(distribution: str):
    """
    Function to decide what distance measure will be used.

    :param distribution: The acronyms (initials) assigned to a distribution finder, which will be used by.
    :return: The choose function.
    """
    if distribution == "CWS":
        return weighted_strategy.weighted_strategy
    elif distribution == "WPS":
        return weighted_strategy.weighted_probability_strategy
    else:
        raise Exception("Distribution not found!")
