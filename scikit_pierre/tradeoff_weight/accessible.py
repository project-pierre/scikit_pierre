"""
This file allows to access all the trade-off weight equations.
"""

from . import weight


def tradeoff_weights_funcs(env_lambda: str):
    """
    Function to decide what tradeoff weight will be used.

    :param env_lambda: The acronyms (initials).
    :return: A float between [0;+inf], which represent the tradeoff weight.
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
    raise NameError(f"Tradeoff weight not found! {env_lambda}")
