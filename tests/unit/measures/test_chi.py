"""
Unit Test Case to the Chi Family measure.
"""

import unittest
from math import sqrt

from ....scikit_pierre.measures import chi


class TestChi(unittest.TestCase):
    """
    Unit Test Case classes to the Chi Family measure.
    """

    def test_squared_euclidean(self):
        """
        This method is to test the Squared Euclidean measure.
        """
        answer = sum(
            [(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2, (0.625 - 0.5) ** 2,
             (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        self.assertEqual(chi.squared_euclidean(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                               q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_person_chi_square(self):
        """
        This method is to test the Pearson Chi Square measure.
        """
        answer_num = sum(
            [((0.389 - 0.35) ** 2) / 0.35, ((0.5 - 0.563) ** 2) / 0.563, ((0.25 - 0.4) ** 2) / 0.4,
             ((0.625 - 0.5) ** 2) / 0.5,
             ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
             ((0.25 - 0.0) ** 2) / 0.00001]
        )
        self.assertEqual(chi.person_chi_square(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                               q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_neyman_square(self):
        """
        This method is to test the Neyman Square measure.
        """
        answer_num = sum(
            [((0.389 - 0.35) ** 2) / 0.389, ((0.5 - 0.563) ** 2) / 0.5, ((0.25 - 0.4) ** 2) / 0.25,
             ((0.625 - 0.5) ** 2) / 0.625,
             ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
             ((0.25 - 0.0) ** 2) / 0.25])
        self.assertEqual(chi.neyman_square(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                           q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_squared_chi_square(self):
        """
        This method is to test the Squared Chi Square measure.
        """
        answer_num = sum(
            [((0.389 - 0.35) ** 2) / (0.389 + 0.35), ((0.5 - 0.563) ** 2) / (0.5 + 0.563),
             ((0.25 - 0.4) ** 2) / (0.25 + 0.4),
             ((0.625 - 0.5) ** 2) / (0.625 + 0.5),
             ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
             ((0.25 - 0.0) ** 2) / (0.25 + 0.0)])
        self.assertEqual(chi.squared_chi_square(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_probabilistic_symmetric_chi_square(self):
        """
        This method is to test the Probabilistic Symmetric Chi Square measure.
        """
        answer_num = 2 * sum(
            [((0.389 - 0.35) ** 2) / (0.389 + 0.35), ((0.5 - 0.563) ** 2) / (0.5 + 0.563),
             ((0.25 - 0.4) ** 2) / (0.25 + 0.4),
             ((0.625 - 0.5) ** 2) / (0.625 + 0.5),
             ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
             ((0.25 - 0.0) ** 2) / (0.25 + 0.0)])
        self.assertEqual(
            chi.probabilistic_symmetric_chi_square(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                   q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
            answer_num)

    def test_divergence(self):
        """
        This method is to test the Divergence measure.
        """
        answer_num = 2 * sum(
            [((0.389 - 0.35) ** 2) / (0.389 + 0.35) ** 2, ((0.5 - 0.563) ** 2) / (0.5 + 0.563) ** 2,
             ((0.25 - 0.4) ** 2) / (0.25 + 0.4) ** 2,
             ((0.625 - 0.5) ** 2) / (0.625 + 0.5) ** 2,
             ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
             ((0.25 - 0.0) ** 2) / (0.25 + 0.0) ** 2])
        self.assertEqual(chi.divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                        q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_clark(self):
        """
        This method is to test the Clark measure.
        """
        answer_num = sqrt(
            sum([(abs(0.389 - 0.35) / (0.389 + 0.35)) ** 2, (abs(0.5 - 0.563) / (0.5 + 0.563)) ** 2,
                 (abs(0.25 - 0.4) / (0.25 + 0.4)) ** 2,
                 (abs(0.625 - 0.5) / (0.625 + 0.5)) ** 2,
                 (0.0 - 0.0) / 0.00001, (0.0 - 0.0) / 0.00001,
                 (abs(0.25 - 0.0) / (0.25 + 0.0)) ** 2]))
        self.assertEqual(chi.clark(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                   q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_additive_symmetric_chi_squared(self):
        """
        This method is to test the Additive Symmetric Chi Squared measure.
        """
        answer_num = sum([(((0.389 - 0.35) ** 2) * (0.389 + 0.35)) / (0.389 * 0.35),
                          (((0.5 - 0.563) ** 2) * (0.5 + 0.563)) / (0.5 * 0.563),
                          (((0.25 - 0.4) ** 2) * (0.25 + 0.4)) / (0.25 * 0.4),
                          (((0.625 - 0.5) ** 2) * (0.625 + 0.5)) / (0.625 * 0.5),
                          (((0.0 - 0.0) ** 2) * (0.0 + 0.0)) / 0.00001,
                          (((0.0 - 0.0) ** 2) * (0.0 + 0.0)) / 0.00001,
                          (((0.25 - 0.0) ** 2) * (0.25 + 0.0)) / 0.00001])
        self.assertEqual(
            chi.additive_symmetric_chi_squared(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                               q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
            answer_num)
