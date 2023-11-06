from math import log2


def absolute_entropy(items: dict) -> dict:
    """
    The Weighted Genre Distribution - (WGD). The reference for this implementation are from:

    -

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


# Weighted Distribution
def weighted_entropy_genre(items: dict) -> dict:
    """
    The Weighted Genre Distribution - (WGD). The reference for this implementation are from:

    -

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute():
        for index, item in items.items():
            for genre, genre_value in item.genres.items():
                ent = -genre_value*log2(genre_value)
                numerator[genre] = numerator.get(genre, 0) + item.score * ent
                denominator[genre] = denominator.get(genre, 0) + item.score

    compute()
    distribution = {g: numerator[g] / denominator[g] for g in numerator}
    return distribution
