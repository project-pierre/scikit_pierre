import unittest
from math import sqrt

from pierre.measures import minkowski


class TestMinkowski(unittest.TestCase):
    def test_city_block(self):
        answer = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                      abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(minkowski.city_block(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                              q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_euclidean(self):
        answer = sqrt(sum([abs(0.389 - 0.35) ** 2, abs(0.5 - 0.563) ** 2, abs(0.25 - 0.4) ** 2,
                           abs(0.625 - 0.5) ** 2,
                           abs(0.0 - 0.0) ** 2, abs(0.0 - 0.0) ** 2, abs(0.25 - 0.0) ** 2]))
        self.assertEqual(minkowski.euclidean(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_cheb(self):
        answer = max([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4),
                      abs(0.625 - 0.5), abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(minkowski.chebyshev(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_minkowski(self):
        answer = sum([abs(0.389 - 0.35) ** 3, abs(0.5 - 0.563) ** 3, abs(0.25 - 0.4) ** 3,
                      abs(0.625 - 0.5) ** 3,
                      abs(0.0 - 0.0) ** 3, abs(0.0 - 0.0) ** 3, abs(0.25 - 0.0) ** 3]) ** (1 / 3)
        self.assertEqual(minkowski.minkowski(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)
