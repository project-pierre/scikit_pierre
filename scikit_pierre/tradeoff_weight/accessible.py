from . import weight


def lambda_choice(dist: list, env_lambda: str) -> float:
    """
    Function to decide what tradeoff weight will be used.

    :param dist: A list composed of float numbers.
    :param env_lambda: The acronyms (initials).
    :return: A float between [0;+inf], which represent the tradeoff weight.
    """
    if env_lambda == "CGR":
        return weight.genre_count(dist_vec=dist)
    elif env_lambda == "VAR":
        return weight.norm_var(dist_vec=dist)
    else:
        raise Exception("Tradeoff weight not found!")


def tradeoff_weights_funcs(env_lambda: str):
    """
    Function to decide what tradeoff weight will be used.

    :param env_lambda: The acronyms (initials).
    :return: A float between [0;+inf], which represent the tradeoff weight.
    """
    if env_lambda == "CGR":
        return weight.genre_count
    elif env_lambda == "VAR":
        return weight.norm_var
    elif env_lambda[:2] == "C@":
        return float(env_lambda.split('@')[1])
    else:
        raise Exception("Tradeoff weight not found!")

