"""
File to allow access all distribution functions.
"""
from . import class_based, entropy_based, time_based, mixed_based, time_slide_window_based


def distributions_funcs(distribution: str):
    """
    Function to decide what distance measure will be used.

    :param distribution: The acronyms (initials) assigned to a distribution finder,
                        which will be used by.
    :return: The choose function.
    """
    if distribution == "CWS":
        return class_based.class_weighted_strategy
    if distribution == "WPS":
        return class_based.weighted_probability_strategy
    if distribution == "PGD":
        return class_based.pure_genre
    if distribution == "PGD_P":
        return class_based.pure_genre_with_probability_property
    if distribution == "TWB":
        return time_based.time_weighted_based
    if distribution == "TWB_P":
        return time_based.time_weighted_based_with_probability_property
    if distribution == "TGD":
        return time_based.time_genre
    if distribution == "TGD_P":
        return time_based.time_genre_with_probability_property
    if distribution == "GLEB":
        return entropy_based.global_local_entropy_based
    if distribution == "GLEB_P":
        return entropy_based.global_local_entropy_based_with_probability_property
    if distribution == "TWB_GLEB":
        return mixed_based.mixed_gleb_twb
    if distribution == "TWB_GLEB_P":
        return mixed_based.mixed_gleb_twb_with_probability_property
    if distribution == "TSW":
        return time_slide_window_based.temporal_slide_window
    if distribution == "TSW_P":
        return time_slide_window_based.temporal_slide_window_with_probability_property
    if distribution == "TSW_GLEB":
        return time_slide_window_based.mixed_tsw_gleb
    if distribution == "TSW_GLEB_P":
        return time_slide_window_based.mixed_tsw_gleb_with_probability_property
    if distribution == "TSW_TWB":
        return time_slide_window_based.mixed_tsw_twb
    if distribution == "TSW_TWB_P":
        return time_slide_window_based.mixed_tsw_twb_with_probability_property
    if distribution == "TSW_TWB_GLEB":
        return time_slide_window_based.mixed_tsw_twb_gleb
    if distribution == "TSW_TWB_GLEB_P":
        return time_slide_window_based.mixed_tsw_twb_gleb_with_probability_property
    raise NameError(f"Distribution not found! {distribution}")
