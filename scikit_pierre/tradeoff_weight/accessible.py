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
    elif env_lambda == "CGR":
        return weight.genre_count
    elif env_lambda == "VAR":
        return weight.norm_var
    elif env_lambda == "STD":
        return weight.norm_std
    elif env_lambda == "TRT":
        return weight.trust
    elif env_lambda == "AMP":
        return weight.amplitude
    elif env_lambda == "EFF":
        return weight.efficiency
    else:
        raise NameError(f"Tradeoff weight not found! {env_lambda}")
