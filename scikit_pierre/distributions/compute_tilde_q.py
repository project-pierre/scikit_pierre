"""
This file presents the distribution tilde q, which deals with the zero problem.
"""


def compute_tilde_q(p: list, q: list, alpha: float = 0.01) -> list:
    """
    Function to compute the tilde q distribution values.

    :param p: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param q: A list with float numbers, which represents the distribution values,
                p and q need to be the same size.
    :param alpha: Trade-off weight value to Realized distribution \tilde{q}

    :return: A list with floats numbers that represent the new realized distribution values.
    """
    return [(1 - alpha) * j + alpha * i for i, j in zip(p, q)]
