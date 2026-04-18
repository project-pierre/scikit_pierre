"""
Factory accessor for per-user item-class probability distribution functions.

Maps string acronyms to the underlying callables.  Each distribution
function accepts a ``dict`` of :class:`~scikit_pierre.models.item.Item`
instances and returns a ``dict`` mapping genre/class label to a float value.
"""
from . import class_based, entropy_based, time_based, mixed_based, time_slide_window_based


def distributions_funcs(  # pylint: disable=too-many-return-statements,too-many-branches
        distribution: str):
    """
    Return the distribution function identified by *distribution*.

    Parameters
    ----------
    distribution : str
        Acronym for the desired distribution strategy.  Supported values:

        - ``"CWS"``          — Class Weighted Strategy
        - ``"WPS"``          — Weighted Probability Strategy
        - ``"PGD"``          — Pure Genre Distribution
        - ``"PGD_P"``        — Pure Genre Distribution (probability-normalised)
        - ``"TWB"``          — Time Weight Based
        - ``"TWB_P"``        — Time Weight Based (probability-normalised)
        - ``"TGD"``          — Time Genre Distribution
        - ``"TGD_P"``        — Time Genre Distribution (probability-normalised)
        - ``"GLEB"``         — Global-Local Entropy Based
        - ``"GLEB_P"``       — Global-Local Entropy Based (probability-normalised)
        - ``"TWB_GLEB"``     — Time Weight + Entropy Based
        - ``"TWB_GLEB_P"``   — Time Weight + Entropy Based (probability-normalised)
        - ``"TSW"``          — Temporal Slide Window
        - ``"TSW_P"``        — Temporal Slide Window (probability-normalised)
        - ``"TSW_GLEB"``     — Temporal Slide Window + Entropy
        - ``"TSW_GLEB_P"``   — Temporal Slide Window + Entropy (probability-normalised)
        - ``"TSW_TWB"``      — Temporal Slide Window + Time Weight
        - ``"TSW_TWB_P"``    — Temporal Slide Window + Time Weight (probability-normalised)
        - ``"TSW_TWB_GLEB"`` — Temporal Slide Window + Time Weight + Entropy
        - ``"TSW_TWB_GLEB_P"`` — all combined (probability-normalised)

    Returns
    -------
    callable
        A function with signature ``(items: dict) -> dict``.

    Raises
    ------
    NameError
        If *distribution* does not match any known key.
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
