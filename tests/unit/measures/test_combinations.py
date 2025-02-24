"""
Unit Test Case to the Chi Family measure.
"""

import unittest
from math import log, sqrt

from ....scikit_pierre.measures import combinations


class TestCombinations(unittest.TestCase):
    """
    Unit Test Case classes to the Chi Family measure.
    """

    def test_taneja(self):
        """
        This method is to test the Squared Euclidean measure.
        """
        answer = sum([
            ((0.389 + 0.35) / 2) * log((0.389 + 0.35) / (2 * sqrt(0.389 * 0.35))),
            ((0.5 + 0.563) / 2) * log((0.5 + 0.563) / (2 * sqrt(0.5 * 0.563))),
            ((0.25 + 0.4) / 2) * log((0.25 + 0.4) / (2 * sqrt(0.25 * 0.4))),
            ((0.625 + 0.5) / 2) * log((0.625 + 0.5) / (2 * sqrt(0.625 * 0.5))),
            ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / (2 * sqrt(0.00001 * 0.00001))),
            ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / (2 * sqrt(0.00001 * 0.00001))),
            ((0.25 + 0.00001) / 2) * log((0.25 + 0.00001) / (2 * sqrt(0.25 * 0.00001))),
        ])
        self.assertEqual(combinations.taneja(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_kumar_johnson(self):
        answer = sum([
            ((((0.389 ** 2) - (0.35 ** 2)) ** 2) / (2 * ((0.389 * 0.35) ** (3 / 2)))),
            ((((0.5 ** 2) - (0.563 ** 2)) ** 2) / (2 * ((0.5 * 0.563) ** (3 / 2)))),
            ((((0.25 ** 2) - (0.4 ** 2)) ** 2) / (2 * ((0.25 * 0.4) ** (3 / 2)))),
            ((((0.625 ** 2) - (0.5 ** 2)) ** 2) / (2 * ((0.625 * 0.5) ** (3 / 2)))),
            ((((0.00001 ** 2) - (0.00001 ** 2)) ** 2) / (2 * ((0.00001 * 0.00001) ** (3 / 2)))),
            ((((0.00001 ** 2) - (0.00001 ** 2)) ** 2) / (2 * ((0.00001 * 0.00001) ** (3 / 2)))),
            ((((0.25 ** 2) - (0.00001 ** 2)) ** 2) / (2 * ((0.25 * 0.00001) ** (3 / 2)))),
        ])
        self.assertEqual(combinations.kumar_johnson(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                    q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_avg(self):
        maxc = max([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                    abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer = sum([
            abs(0.389 - 0.35) + maxc, abs(0.5 - 0.563) + maxc, abs(0.25 - 0.4) + maxc,
            abs(0.625 - 0.5) + maxc,
            abs(0.00001 - 0.00001) + maxc, abs(0.00001 - 0.00001) + maxc, abs(0.25 - 0.00001) + maxc
        ])
        self.assertEqual(combinations.avg(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                          q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer / 2)

    def test_weighted_total_variation(self):
        answer = sum([(0.389 + 1) * abs(0.389 - 0.35), (0.5 + 1) * abs(0.5 - 0.563),
                      (0.25 + 1) * abs(0.25 - 0.4),
                      (0.625 + 1) * abs(0.625 - 0.5), (0.00001 + 1) * abs(0.00001 - 0.00001),
                      (0.00001 + 1) * abs(0.00001 - 0.00001), (0.25 + 1) * abs(0.25 - 0.00001)])
        self.assertEqual(
            combinations.weighted_total_variation(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                  q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
            answer / 2)
