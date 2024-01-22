from . import class_based, entropy_based, time_based, mixed_based, time_slide_window_based
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
    elif distribution == "PGD":
        return class_based.pure_genre
    elif distribution == "PGD_P":
        return class_based.pure_genre_with_probability_property
    elif distribution == "TWB":
        return time_based.time_weighted_based
    elif distribution == "TWB_P":
        return time_based.time_weighted_based_with_probability_property
    elif distribution == "TGD":
        return time_based.time_genre
    elif distribution == "TGD_P":
        return time_based.time_genre_with_probability_property
    elif distribution == "GLEB":
        return entropy_based.global_local_entropy_based
    elif distribution == "GLEB_P":
        return entropy_based.global_local_entropy_based_with_probability_property
    elif distribution == "GLEB_TWB":
        return mixed_based.mixed_gleb_twb
    elif distribution == "GLEB_TWB_P":
        return mixed_based.mixed_gleb_twb_with_probability_property
    elif distribution == "TSW":
        return time_slide_window_based.temporal_slide_window
    elif distribution == "TSW_P":
        return time_slide_window_based.temporal_slide_window_with_probability_property
    elif distribution == "TSW_GLEB":
        return time_slide_window_based.mixed_tsw_gleb
    elif distribution == "TSW_GLEB_P":
        return time_slide_window_based.mixed_tsw_gleb_with_probability_property
    elif distribution == "TSW_TWB":
        return time_slide_window_based.mixed_tsw_twb
    elif distribution == "TSW_TWB_P":
        return time_slide_window_based.mixed_tsw_twb_with_probability_property
    elif distribution == "TSW_TWB_GLEB":
        return time_slide_window_based.mixed_tsw_twb_gleb
    elif distribution == "TSW_TWB_GLEB_P":
        return time_slide_window_based.mixed_tsw_twb_gleb_with_probability_property
    else:
        raise Exception(f"Distribution not found! {distribution}")


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
    elif distribution == "TWB":
        return weighted_strategy.time_weighted_based
    else:
        raise Exception("Distribution not found!")
