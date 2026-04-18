"""
Unit Test Cases for the Combinations Family measures.
"""
import unittest
from math import log, sqrt

from scikit_pierre.measures import combinations

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestCombinations(unittest.TestCase):

    # ── taneja ────────────────────────────────────────────────────────────

    def test_taneja(self):
        answer = sum([
            ((0.389 + 0.35) / 2) * log((0.389 + 0.35) / (2 * sqrt(0.389 * 0.35))),
            ((0.5 + 0.563) / 2) * log((0.5 + 0.563) / (2 * sqrt(0.5 * 0.563))),
            ((0.25 + 0.4) / 2) * log((0.25 + 0.4) / (2 * sqrt(0.25 * 0.4))),
            ((0.625 + 0.5) / 2) * log((0.625 + 0.5) / (2 * sqrt(0.625 * 0.5))),
            ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / (2 * sqrt(0.00001 * 0.00001))),
            ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / (2 * sqrt(0.00001 * 0.00001))),
            ((0.25 + 0.00001) / 2) * log((0.25 + 0.00001) / (2 * sqrt(0.25 * 0.00001))),
        ])
        self.assertEqual(combinations.taneja(p=P_STD, q=Q_STD), answer)

    def test_taneja_identical_distributions(self):
        self.assertEqual(combinations.taneja(P_NZ, P_NZ), 0.0)

    def test_taneja_single_identical(self):
        self.assertEqual(combinations.taneja([0.5], [0.5]), 0.0)

    def test_taneja_single_different(self):
        p_a, q_b = 0.3, 0.7
        answer = ((p_a + q_b) / 2) * log((p_a + q_b) / (2 * sqrt(p_a * q_b)))
        self.assertEqual(combinations.taneja([p_a], [q_b]), answer)

    def test_taneja_symmetric(self):
        self.assertEqual(combinations.taneja(P_NZ, Q_NZ),
                         combinations.taneja(Q_NZ, P_NZ))

    def test_taneja_non_negative(self):
        self.assertGreaterEqual(combinations.taneja(P_STD, Q_STD), 0.0)

    def test_taneja_zeros_use_epsilon(self):
        answer = ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / (2 * sqrt(0.00001 * 0.00001)))
        self.assertEqual(combinations.taneja([0.0], [0.0]), answer)

    def test_taneja_uniform_distributions(self):
        self.assertEqual(combinations.taneja(P_UNI, Q_UNI), 0.0)

    def test_taneja_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sum([
            ((p_i + q_i) / 2) * log((p_i + q_i) / (2 * sqrt(p_i * q_i)))
            for p_i, q_i in zip(p, q)
        ])
        self.assertEqual(combinations.taneja(p, q), answer)

    def test_taneja_p_zero_uses_epsilon(self):
        p_a, q_b = 0.00001, 0.5
        answer = ((p_a + q_b) / 2) * log((p_a + q_b) / (2 * sqrt(p_a * q_b)))
        self.assertEqual(combinations.taneja([0.0], [0.5]), answer)

    def test_taneja_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum([
            ((p_i + q_i) / 2) * log((p_i + q_i) / (2 * sqrt(p_i * q_i)))
            for p_i, q_i in zip(p, q)
        ])
        self.assertEqual(combinations.taneja(p, q), answer)

    # ── kumar_johnson ────────────────────────────────────────────────────

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
        self.assertEqual(combinations.kumar_johnson(p=P_STD, q=Q_STD), answer)

    def test_kumar_johnson_identical_distributions(self):
        self.assertEqual(combinations.kumar_johnson(P_NZ, P_NZ), 0.0)

    def test_kumar_johnson_single_identical(self):
        self.assertEqual(combinations.kumar_johnson([0.5], [0.5]), 0.0)

    def test_kumar_johnson_single_different(self):
        p_a, q_b = 0.3, 0.7
        answer = ((p_a ** 2 - q_b ** 2) ** 2) / (2 * (p_a * q_b) ** (3 / 2))
        self.assertEqual(combinations.kumar_johnson([p_a], [q_b]), answer)

    def test_kumar_johnson_symmetric(self):
        self.assertEqual(combinations.kumar_johnson(P_NZ, Q_NZ),
                         combinations.kumar_johnson(Q_NZ, P_NZ))

    def test_kumar_johnson_non_negative(self):
        self.assertGreaterEqual(combinations.kumar_johnson(P_STD, Q_STD), 0.0)

    def test_kumar_johnson_zeros_use_epsilon(self):
        answer = ((0.00001 ** 2 - 0.00001 ** 2) ** 2) / (2 * (0.00001 * 0.00001) ** (3 / 2))
        self.assertEqual(combinations.kumar_johnson([0.0], [0.0]), answer)

    def test_kumar_johnson_uniform_distributions(self):
        self.assertEqual(combinations.kumar_johnson(P_UNI, Q_UNI), 0.0)

    def test_kumar_johnson_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sum([
            ((p_i ** 2 - q_i ** 2) ** 2) / (2 * (p_i * q_i) ** (3 / 2))
            for p_i, q_i in zip(p, q)
        ])
        self.assertEqual(combinations.kumar_johnson(p, q), answer)

    def test_kumar_johnson_p_zero_uses_epsilon(self):
        p_a, q_b = 0.00001, 0.5
        answer = ((p_a ** 2 - q_b ** 2) ** 2) / (2 * (p_a * q_b) ** (3 / 2))
        self.assertEqual(combinations.kumar_johnson([0.0], [0.5]), answer)

    def test_kumar_johnson_large_values(self):
        p_a, q_b = 10.0, 15.0
        answer = ((p_a ** 2 - q_b ** 2) ** 2) / (2 * (p_a * q_b) ** (3 / 2))
        self.assertEqual(combinations.kumar_johnson([p_a], [q_b]), answer)

    # ── avg ───────────────────────────────────────────────────────────────

    def test_avg(self):
        maxc = max([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                    abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer = sum([
            abs(0.389 - 0.35) + maxc, abs(0.5 - 0.563) + maxc, abs(0.25 - 0.4) + maxc,
            abs(0.625 - 0.5) + maxc,
            abs(0.00001 - 0.00001) + maxc, abs(0.00001 - 0.00001) + maxc,
            abs(0.25 - 0.00001) + maxc
        ])
        self.assertEqual(combinations.avg(p=P_STD, q=Q_STD), answer / 2)

    def test_avg_identical_distributions(self):
        result = combinations.avg(P_NZ, P_NZ)
        maxc = 0.0
        expected = sum([abs(v - v) + maxc for v in P_NZ]) / 2
        self.assertEqual(result, expected)

    def test_avg_single_identical(self):
        result = combinations.avg([0.5], [0.5])
        self.assertEqual(result, 0.0)

    def test_avg_single_different(self):
        p_a, q_b = 0.3, 0.7
        diff = abs(p_a - q_b)
        answer = (diff + diff) / 2
        self.assertEqual(combinations.avg([p_a], [q_b]), answer)

    def test_avg_non_negative(self):
        self.assertGreaterEqual(combinations.avg(P_STD, Q_STD), 0.0)

    def test_avg_symmetric(self):
        self.assertEqual(combinations.avg(P_NZ, Q_NZ),
                         combinations.avg(Q_NZ, P_NZ))

    def test_avg_zeros_use_epsilon(self):
        result = combinations.avg([0.0], [0.0])
        diff = abs(0.00001 - 0.00001)
        expected = (diff + diff) / 2
        self.assertEqual(result, expected)

    def test_avg_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        maxc = max(abs(0.3 - 0.4), abs(0.7 - 0.6))
        answer = (abs(0.3 - 0.4) + maxc + abs(0.7 - 0.6) + maxc) / 2
        self.assertEqual(combinations.avg(p, q), answer)

    def test_avg_uniform_distributions(self):
        result = combinations.avg(P_UNI, Q_UNI)
        self.assertEqual(result, 0.0)

    def test_avg_large_difference(self):
        p, q = [0.0, 1.0], [1.0, 0.0]
        maxc = max(abs(0.0 - 1.0), abs(1.0 - 0.0))
        answer = (abs(0.00001 - 1.0) + maxc + abs(1.0 - 0.00001) + maxc) / 2
        self.assertEqual(combinations.avg(p, q), answer)

    def test_avg_p_zero_uses_epsilon(self):
        result = combinations.avg([0.0], [0.5])
        maxc = abs(0.0 - 0.5)
        expected = (abs(0.00001 - 0.5) + maxc) / 2
        self.assertEqual(result, expected)

    # ── weighted_total_variation ──────────────────────────────────────────

    def test_weighted_total_variation(self):
        answer = sum([
            (0.389 + 1) * abs(0.389 - 0.35), (0.5 + 1) * abs(0.5 - 0.563),
            (0.25 + 1) * abs(0.25 - 0.4), (0.625 + 1) * abs(0.625 - 0.5),
            (0.00001 + 1) * abs(0.00001 - 0.00001), (0.00001 + 1) * abs(0.00001 - 0.00001),
            (0.25 + 1) * abs(0.25 - 0.00001)
        ])
        self.assertEqual(combinations.weighted_total_variation(p=P_STD, q=Q_STD), answer / 2)

    def test_weighted_total_variation_identical(self):
        self.assertEqual(combinations.weighted_total_variation(P_NZ, P_NZ), 0.0)

    def test_weighted_total_variation_single_identical(self):
        self.assertEqual(combinations.weighted_total_variation([0.5], [0.5]), 0.0)

    def test_weighted_total_variation_single_different(self):
        p_a, q_b = 0.3, 0.7
        answer = (p_a + 1) * abs(p_a - q_b) / 2
        self.assertEqual(combinations.weighted_total_variation([p_a], [q_b]), answer)

    def test_weighted_total_variation_non_negative(self):
        self.assertGreaterEqual(combinations.weighted_total_variation(P_STD, Q_STD), 0.0)

    def test_weighted_total_variation_zeros_use_epsilon(self):
        answer = (0.00001 + 1) * abs(0.00001 - 0.00001) / 2
        self.assertEqual(combinations.weighted_total_variation([0.0], [0.0]), answer)

    def test_weighted_total_variation_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = ((0.3 + 1) * abs(0.3 - 0.4) + (0.7 + 1) * abs(0.7 - 0.6)) / 2
        self.assertEqual(combinations.weighted_total_variation(p, q), answer)

    def test_weighted_total_variation_uniform(self):
        self.assertEqual(combinations.weighted_total_variation(P_UNI, Q_UNI), 0.0)

    def test_weighted_total_variation_p_zero_uses_epsilon(self):
        answer = (0.00001 + 1) * abs(0.00001 - 0.5) / 2
        self.assertEqual(combinations.weighted_total_variation([0.0], [0.5]), answer)

    def test_weighted_total_variation_large_values(self):
        p_a, q_b = 10.0, 15.0
        answer = (p_a + 1) * abs(p_a - q_b) / 2
        self.assertEqual(combinations.weighted_total_variation([p_a], [q_b]), answer)

    def test_weighted_total_variation_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum([(p_i + 1) * abs(p_i - q_i) for p_i, q_i in zip(p, q)]) / 2
        self.assertEqual(combinations.weighted_total_variation(p, q), answer)
