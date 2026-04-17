"""
Temporal sliding-window distribution functions.

These functions split the user's interaction history into time windows,
compute a base distribution per window, and then average the results.
This captures gradual preference drift without discarding older data entirely.

When the number of interactions is small (≤ ``2 * major`` windows), the
base distribution is applied to the full item set without windowing.

All public functions accept a ``dict`` of
:class:`~scikit_pierre.models.item.Item` instances and return a ``dict``
mapping genre label to a float.
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
    Core implementation of the Temporal Slide Window (TSW) distribution.

    Splits the item set into ``2 * major`` windows, computes the base
    distribution for each window (plus one combined boundary window), and
    returns the mean per-genre value across all windows.  Falls back to a
    single-window computation when ``len(items) <= 2 * major``.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    major : int, optional
        Number of primary windows.  Total windows = ``2 * major``.
        Defaults to 10.
    using : str, optional
        Acronym of the base distribution to apply within each window.
        Supported values: ``"CWS"`` (default), ``"GLEB"``, ``"TWB"``,
        ``"GLEB_TWB"``.

    Returns
    -------
    dict
        Mapping of genre label -> mean distribution value across windows.
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
                distribution_list.append(
                    base_distribution(dict({item for _, item in enum_items[floor:]})))
            else:
                distribution_list.append(
                    base_distribution(dict(item for _, item in enum_items[floor:top])))
            floor = (i - 1) * batch_size

        distribution_list.append(base_distribution({
            **dict(item for _, item in enum_items[:batch_size]),
            **dict(item for _, item in enum_items[floor:])
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


def temporal_slide_window_base_function_with_probability_property(
        items: dict, using: str = "CWS"
) -> dict:
    """
    Core implementation of TSW normalised to sum to 1.0.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Acronym of the base distribution.  See
        :func:`temporal_slide_window_base_function`.

    Returns
    -------
    dict
        Normalised TSW genre probability distribution summing to 1.0.
    """
    distribution = temporal_slide_window_base_function(items=items, using=using)
    total = sum(distribution.values())
    final_distribution = {g: value / total for g, value in distribution.items()}
    return final_distribution


def temporal_slide_window(items: dict, using: str = "CWS") -> dict:
    """
    Compute the Temporal Slide Window (TSW) distribution using CWS windows.

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"CWS"``.

    Returns
    -------
    dict
        Mapping of genre label -> mean TSW distribution value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def temporal_slide_window_with_probability_property(items: dict, using: str = "CWS") -> dict:
    """
    Compute the TSW distribution normalised to sum to 1.0 (TSW_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"CWS"``.

    Returns
    -------
    dict
        Normalised TSW genre probability distribution summing to 1.0.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_gleb(items: dict, using: str = "GLEB") -> dict:
    """
    Compute the TSW distribution using GLEB windows (TSW_GLEB).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"GLEB"``.

    Returns
    -------
    dict
        Mapping of genre label -> mean TSW_GLEB distribution value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_gleb_with_probability_property(items: dict, using: str = "GLEB") -> dict:
    """
    Compute the TSW_GLEB distribution normalised to sum to 1.0 (TSW_GLEB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"GLEB"``.

    Returns
    -------
    dict
        Normalised TSW_GLEB genre probability distribution summing to 1.0.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_twb(items: dict, using: str = "TWB") -> dict:
    """
    Compute the TSW distribution using TWB windows (TSW_TWB).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"TWB"``.

    Returns
    -------
    dict
        Mapping of genre label -> mean TSW_TWB distribution value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_twb_with_probability_property(items: dict, using: str = "TWB") -> dict:
    """
    Compute the TSW_TWB distribution normalised to sum to 1.0 (TSW_TWB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"TWB"``.

    Returns
    -------
    dict
        Normalised TSW_TWB genre probability distribution summing to 1.0.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)


def mixed_tsw_twb_gleb(items: dict, using: str = "GLEB_TWB") -> dict:
    """
    Compute the TSW distribution using combined GLEB+TWB windows (TSW_TWB_GLEB).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"GLEB_TWB"``.

    Returns
    -------
    dict
        Mapping of genre label -> mean TSW_TWB_GLEB distribution value.
    """
    return temporal_slide_window_base_function(items=items, using=using)


def mixed_tsw_twb_gleb_with_probability_property(items: dict, using: str = "GLEB_TWB") -> dict:
    """
    Compute the TSW_TWB_GLEB distribution normalised to sum to 1.0 (TSW_TWB_GLEB_P).

    Parameters
    ----------
    items : dict
        Mapping of item_id -> :class:`~scikit_pierre.models.item.Item`.
    using : str, optional
        Base distribution acronym.  Defaults to ``"GLEB_TWB"``.

    Returns
    -------
    dict
        Normalised TSW_TWB_GLEB genre probability distribution summing to 1.0.
    """
    return temporal_slide_window_base_function_with_probability_property(items=items, using=using)
