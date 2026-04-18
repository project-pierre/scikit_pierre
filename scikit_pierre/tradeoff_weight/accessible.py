"""
Factory accessor for trade-off weight (lambda) functions.

Maps string acronyms — or constant-value strings like ``"C@0.5"`` — to
the corresponding callable from :mod:`weight`.
"""

from . import weight


def tradeoff_weights_funcs(env_lambda: str):  # pylint: disable=too-many-return-statements
    """
    Return the trade-off weight function or constant identified by *env_lambda*.

    Parameters
    ----------
    env_lambda : str
        Acronym or constant specification.  Supported values:

        - ``"C@<value>"`` — constant lambda equal to ``float(value)``
          (e.g. ``"C@0.5"`` returns the float ``0.5`` directly).
        - ``"CGR"`` — genre count ratio (:func:`weight.genre_count`)
        - ``"VAR"`` — normalised variance (:func:`weight.norm_var`)
        - ``"STD"`` — normalised standard deviation (:func:`weight.norm_std`)
        - ``"TRT"`` — mean trust (:func:`weight.trust`)
        - ``"AMP"`` — amplitude (:func:`weight.amplitude`)
        - ``"EFF"`` — efficiency / coefficient of variation
          (:func:`weight.efficiency`)
        - ``"MIT"`` — mitigation (:func:`weight.mitigation`)

    Returns
    -------
    float or callable
        A float when ``env_lambda`` starts with ``"C@"``; otherwise a
        callable with signature ``(dist_vec: list, ...) -> float``.

    Raises
    ------
    NameError
        If *env_lambda* does not match any known key.
    """
    if env_lambda[:2] == "C@":
        return float(env_lambda.split('@')[1])
    if env_lambda == "CGR":
        return weight.genre_count
    if env_lambda == "VAR":
        return weight.norm_var
    if env_lambda == "STD":
        return weight.norm_std
    if env_lambda == "TRT":
        return weight.trust
    if env_lambda == "AMP":
        return weight.amplitude
    if env_lambda == "EFF":
        return weight.efficiency
    if env_lambda == "MIT":
        return weight.mitigation
    raise NameError(f"Tradeoff weight not found! {env_lambda}")
