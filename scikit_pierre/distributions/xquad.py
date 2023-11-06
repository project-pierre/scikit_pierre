import math
from math import log2


def xquad(item: object, recommendations: dict, p: dict, q: dict):
    """
    """
    aspect = []

    def compute_aspect_probability(a):
        return p[a]*q[a] * math.prod([1 - q[j] for j in recommendations])

    return sum([compute_aspect_probability(a) for a in aspect])


