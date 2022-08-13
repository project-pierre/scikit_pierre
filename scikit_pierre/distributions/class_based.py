def class_weighted_strategy(items: dict) -> dict:
    """
    The Class Weighted Strategy - (CWS). The reference for this implementation are from:

    - Silva et. al. (2021). https://doi.org/10.1016/j.eswa.2021.115112

    - Kaya and Bridge (2019). https://doi.org/10.1145/3298689.3347045

    - Steck (2018). https://doi.org/10.1145/3240323.3240372

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    numerator = {}
    denominator = {}

    def compute():
        for index, item in items.items():
            for genre, genre_value in item.classes.items():
                numerator[genre] = numerator.get(genre, 0) + item.score * genre_value
                denominator[genre] = denominator.get(genre, 0) + item.score

    compute()
    distribution = {g: numerator[g] / denominator[g] for g in numerator}
    return distribution


def weighted_probability_strategy(items: dict) -> dict:
    """
    The Weighted Probability Strategy - (WPS). The reference for this implementation are from:

    - Silva and Durão (2022). https://arxiv.org/abs/2204.03706

    :param items: A Dict of Item Class instances.
    :return: A Dict of genre and value.
    """
    distribution = class_weighted_strategy(items)
    total = sum([value for g, value in distribution.items()])
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution
