# ############################################################################################### #
# ######################################### Time Based ########################################## #
# ############################################################################################### #


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
            for category, genre_value in item.classes.items():
                numerator[category] = numerator.get(category, 0) + item.time * item.score * genre_value
                denominator[category] = denominator.get(category, 0) + item.score

    def genre(g):
        if (g in denominator.keys() and denominator[g] > 0.0) and (g in numerator.keys() and numerator[g] > 0.0):
            return numerator[g] / denominator[g]
        else:
            return 0.00001

    compute()
    distribution = {g: genre(g) for g in numerator}
    return distribution


def time_weighted_based_with_probability_property(items: dict) -> dict:
    """
    The Time Weight Based with Probability Property - (TWB_P). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    distribution = time_weighted_based(items)
    total = sum([value for g, value in distribution.items()])
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


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
            for category, genre_value in item.classes.items():
                numerator[category] = numerator.get(category, 0.) + item.time * genre_value
                denominator[category] = denominator.get(category, 0.) + item.time

    compute()
    distribution = {g: numerator[g] / denominator[g] for g in numerator}
    return distribution


def time_genre_with_probability_property(items: dict) -> dict:
    """
    The Time Genre Distribution with Probability Property - (TGD_P). The reference for this implementation are from:

    - <In process>

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    dist = time_genre(items)
    norm = sum(dist.values())
    distribution = {g: value / norm for g, value in dist.items()}
    return distribution


# ############################################################################################### #
# ######################################### Unrevised ########################################### #
# ############################################################################################### #
