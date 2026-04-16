"""
Unit Test Cases for the Shannon Family measures.
"""
import unittest
from math import log

from scikit_pierre.measures import shannon

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestShannon(unittest.TestCase):

    # ── kullback_leibler ──────────────────────────────────────────────────

    def test_kullback_leibler(self):
        answer = sum([0.389 * log(0.389 / 0.35), 0.5 * log(0.5 / 0.563),
                      0.25 * log(0.25 / 0.4), 0.625 * log(0.625 / 0.5),
                      0.00001 * log(0.00001 / 0.00001), 0.00001 * log(0.00001 / 0.00001),
                      0.25 * log(0.25 / 0.00001)])
        self.assertEqual(shannon.kullback_leibler(p=P_STD, q=Q_STD), answer)

    def test_kullback_leibler_identical(self):
        self.assertEqual(shannon.kullback_leibler(P_NZ, P_NZ), 0.0)

    def test_kullback_leibler_single_identical(self):
        self.assertEqual(shannon.kullback_leibler([0.5], [0.5]), 0.0)

    def test_kullback_leibler_single_different(self):
        answer = 0.3 * log(0.3 / 0.7)
        self.assertEqual(shannon.kullback_leibler([0.3], [0.7]), answer)

    def test_kullback_leibler_both_zero_uses_epsilon(self):
        result = shannon.kullback_leibler([0.0], [0.0])
        expected = 0.00001 * log(0.00001 / 0.00001)
        self.assertEqual(result, expected)

    def test_kullback_leibler_asymmetric(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        self.assertNotEqual(shannon.kullback_leibler(p, q), shannon.kullback_leibler(q, p))

    def test_kullback_leibler_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 0.3 * log(0.3 / 0.4) + 0.7 * log(0.7 / 0.6)
        self.assertEqual(shannon.kullback_leibler(p, q), answer)

    def test_kullback_leibler_uniform(self):
        self.assertEqual(shannon.kullback_leibler(P_UNI, Q_UNI), 0.0)

    def test_kullback_leibler_p_zero_uses_epsilon(self):
        result = shannon.kullback_leibler([0.0], [0.5])
        expected = 0.00001 * log(0.00001 / 0.5)
        self.assertEqual(result, expected)

    def test_kullback_leibler_q_zero_uses_epsilon(self):
        result = shannon.kullback_leibler([0.5], [0.0])
        expected = 0.5 * log(0.5 / 0.00001)
        self.assertEqual(result, expected)

    def test_kullback_leibler_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum(p_i * log(p_i / q_i) for p_i, q_i in zip(p, q))
        self.assertEqual(shannon.kullback_leibler(p, q), answer)

    # ── jeffreys ──────────────────────────────────────────────────────────

    def test_jeffreys(self):
        answer = sum([
            (0.389 - 0.35) * log(0.389 / 0.35), (0.5 - 0.563) * log(0.5 / 0.563),
            (0.25 - 0.4) * log(0.25 / 0.4), (0.625 - 0.5) * log(0.625 / 0.5),
            (0.00001 - 0.00001) * log(0.00001 / 0.00001),
            (0.00001 - 0.00001) * log(0.00001 / 0.00001),
            (0.25 - 0.00001) * log(0.25 / 0.00001)
        ])
        self.assertEqual(shannon.jeffreys(p=P_STD, q=Q_STD), answer)

    def test_jeffreys_identical(self):
        self.assertEqual(shannon.jeffreys(P_NZ, P_NZ), 0.0)

    def test_jeffreys_single_identical(self):
        self.assertEqual(shannon.jeffreys([0.5], [0.5]), 0.0)

    def test_jeffreys_single_different(self):
        answer = (0.3 - 0.7) * log(0.3 / 0.7)
        self.assertEqual(shannon.jeffreys([0.3], [0.7]), answer)

    def test_jeffreys_symmetric(self):
        self.assertEqual(shannon.jeffreys(P_NZ, Q_NZ), shannon.jeffreys(Q_NZ, P_NZ))

    def test_jeffreys_non_negative(self):
        self.assertGreaterEqual(shannon.jeffreys(P_NZ, Q_NZ), 0.0)

    def test_jeffreys_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) * log(0.3 / 0.4) + (0.7 - 0.6) * log(0.7 / 0.6)
        self.assertEqual(shannon.jeffreys(p, q), answer)

    def test_jeffreys_uniform(self):
        self.assertEqual(shannon.jeffreys(P_UNI, Q_UNI), 0.0)

    def test_jeffreys_both_zero_uses_epsilon(self):
        result = shannon.jeffreys([0.0], [0.0])
        expected = (0.00001 - 0.00001) * log(0.00001 / 0.00001)
        self.assertEqual(result, expected)

    def test_jeffreys_equals_sum_of_kl(self):
        kl_pq = shannon.kullback_leibler(P_NZ, Q_NZ)
        kl_qp = shannon.kullback_leibler(Q_NZ, P_NZ)
        jef = shannon.jeffreys(P_NZ, Q_NZ)
        self.assertAlmostEqual(jef, kl_pq + kl_qp, places=10)

    def test_jeffreys_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum((p_i - q_i) * log(p_i / q_i) for p_i, q_i in zip(p, q))
        self.assertEqual(shannon.jeffreys(p, q), answer)

    # ── k_divergence ──────────────────────────────────────────────────────

    def test_k_divergence(self):
        answer = sum([
            0.389 * log((2 * 0.389) / (0.389 + 0.35)),
            0.5 * log((2 * 0.5) / (0.5 + 0.563)),
            0.25 * log((2 * 0.25) / (0.25 + 0.4)),
            0.625 * log((2 * 0.625) / (0.625 + 0.5)),
            0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
            0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
            0.25 * log((2 * 0.25) / (0.25 + 0.00001))
        ])
        self.assertEqual(shannon.k_divergence(p=P_STD, q=Q_STD), answer)

    def test_k_divergence_identical(self):
        self.assertEqual(shannon.k_divergence(P_NZ, P_NZ), 0.0)

    def test_k_divergence_single_identical(self):
        self.assertEqual(shannon.k_divergence([0.5], [0.5]), 0.0)

    def test_k_divergence_single_different(self):
        answer = 0.3 * log((2 * 0.3) / (0.3 + 0.7))
        self.assertEqual(shannon.k_divergence([0.3], [0.7]), answer)

    def test_k_divergence_both_zero_uses_epsilon(self):
        result = shannon.k_divergence([0.0], [0.0])
        expected = 0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001))
        self.assertEqual(result, expected)

    def test_k_divergence_asymmetric(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        self.assertNotEqual(shannon.k_divergence(p, q), shannon.k_divergence(q, p))

    def test_k_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 * log((2 * 0.3) / (0.3 + 0.4)) +
                  0.7 * log((2 * 0.7) / (0.7 + 0.6)))
        self.assertEqual(shannon.k_divergence(p, q), answer)

    def test_k_divergence_uniform(self):
        self.assertEqual(shannon.k_divergence(P_UNI, Q_UNI), 0.0)

    def test_k_divergence_non_negative_for_identical(self):
        self.assertGreaterEqual(shannon.k_divergence(P_NZ, P_NZ), 0.0)

    def test_k_divergence_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum(p_i * log((2 * p_i) / (p_i + q_i)) for p_i, q_i in zip(p, q))
        self.assertEqual(shannon.k_divergence(p, q), answer)

    def test_k_divergence_q_zero_uses_epsilon(self):
        result = shannon.k_divergence([0.5], [0.0])
        expected = 0.5 * log((2 * 0.5) / (0.5 + 0.00001))
        self.assertEqual(result, expected)

    # ── topsoe ────────────────────────────────────────────────────────────

    def test_topsoe(self):
        answer = sum([
            (0.389 * log((2 * 0.389) / (0.389 + 0.35))) + (0.35 * log((2 * 0.35) / (0.389 + 0.35))),
            (0.5 * log((2 * 0.5) / (0.5 + 0.563))) + (0.563 * log((2 * 0.563) / (0.5 + 0.563))),
            (0.25 * log((2 * 0.25) / (0.25 + 0.4))) + (0.4 * log((2 * 0.4) / (0.25 + 0.4))),
            (0.625 * log((2 * 0.625) / (0.625 + 0.5))) + (0.5 * log((2 * 0.5) / (0.625 + 0.5))),
            (0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001))) + (0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001))),
            (0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001))) + (0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001))),
            (0.25 * log((2 * 0.25) / (0.25 + 0.00001))) + (0.00001 * log((2 * 0.00001) / (0.25 + 0.00001))),
        ])
        self.assertEqual(shannon.topsoe(p=P_STD, q=Q_STD), answer)

    def test_topsoe_identical(self):
        self.assertEqual(shannon.topsoe(P_NZ, P_NZ), 0.0)

    def test_topsoe_single_identical(self):
        self.assertEqual(shannon.topsoe([0.5], [0.5]), 0.0)

    def test_topsoe_single_different(self):
        p_a, q_b = 0.3, 0.7
        answer = (p_a * log((2 * p_a) / (p_a + q_b)) +
                  q_b * log((2 * q_b) / (p_a + q_b)))
        self.assertEqual(shannon.topsoe([p_a], [q_b]), answer)

    def test_topsoe_symmetric(self):
        self.assertEqual(shannon.topsoe(P_NZ, Q_NZ), shannon.topsoe(Q_NZ, P_NZ))

    def test_topsoe_non_negative(self):
        self.assertGreaterEqual(shannon.topsoe(P_NZ, Q_NZ), 0.0)

    def test_topsoe_equals_sum_of_k_divergence(self):
        kpq = shannon.k_divergence(P_NZ, Q_NZ)
        kqp = shannon.k_divergence(Q_NZ, P_NZ)
        topsoe_result = shannon.topsoe(P_NZ, Q_NZ)
        self.assertAlmostEqual(topsoe_result, kpq + kqp, places=10)

    def test_topsoe_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sum([
            (pi * log((2 * pi) / (pi + qi))) + (qi * log((2 * qi) / (pi + qi)))
            for pi, qi in zip(p, q)
        ])
        self.assertEqual(shannon.topsoe(p, q), answer)

    def test_topsoe_uniform(self):
        self.assertEqual(shannon.topsoe(P_UNI, Q_UNI), 0.0)

    def test_topsoe_both_zero_uses_epsilon(self):
        result = shannon.topsoe([0.0], [0.0])
        expected = (0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)) +
                    0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)))
        self.assertEqual(result, expected)

    # ── jensen_shannon ────────────────────────────────────────────────────

    def test_jensen_shannon(self):
        answer_l = sum([0.389 * log((2 * 0.389) / (0.389 + 0.35)),
                        0.5 * log((2 * 0.5) / (0.5 + 0.563)),
                        0.25 * log((2 * 0.25) / (0.25 + 0.4)),
                        0.625 * log((2 * 0.625) / (0.625 + 0.5)),
                        0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
                        0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
                        0.25 * log((2 * 0.25) / (0.25 + 0.00001))])
        answer_r = sum([0.35 * log((2 * 0.35) / (0.389 + 0.35)),
                        0.563 * log((2 * 0.563) / (0.5 + 0.563)),
                        0.4 * log((2 * 0.4) / (0.25 + 0.4)),
                        0.5 * log((2 * 0.5) / (0.625 + 0.5)),
                        0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
                        0.00001 * log((2 * 0.00001) / (0.00001 + 0.00001)),
                        0.00001 * log((2 * 0.00001) / (0.25 + 0.00001))])
        self.assertEqual(shannon.jensen_shannon(p=P_STD, q=Q_STD), (1 / 2) * (answer_l + answer_r))

    def test_jensen_shannon_identical(self):
        self.assertEqual(shannon.jensen_shannon(P_NZ, P_NZ), 0.0)

    def test_jensen_shannon_single_identical(self):
        self.assertEqual(shannon.jensen_shannon([0.5], [0.5]), 0.0)

    def test_jensen_shannon_single_different(self):
        p_a, q_b = 0.3, 0.7
        l_part = p_a * log((2 * p_a) / (p_a + q_b))
        r_part = q_b * log((2 * q_b) / (p_a + q_b))
        answer = 0.5 * (l_part + r_part)
        self.assertEqual(shannon.jensen_shannon([p_a], [q_b]), answer)

    def test_jensen_shannon_symmetric(self):
        self.assertEqual(shannon.jensen_shannon(P_NZ, Q_NZ),
                         shannon.jensen_shannon(Q_NZ, P_NZ))

    def test_jensen_shannon_non_negative(self):
        self.assertGreaterEqual(shannon.jensen_shannon(P_NZ, Q_NZ), 0.0)

    def test_jensen_shannon_equals_half_topsoe(self):
        js = shannon.jensen_shannon(P_NZ, Q_NZ)
        ts = shannon.topsoe(P_NZ, Q_NZ)
        self.assertAlmostEqual(js, ts / 2, places=10)

    def test_jensen_shannon_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        l_part = sum(pi * log((2 * pi) / (pi + qi)) for pi, qi in zip(p, q))
        r_part = sum(qi * log((2 * qi) / (pi + qi)) for pi, qi in zip(p, q))
        answer = 0.5 * (l_part + r_part)
        self.assertEqual(shannon.jensen_shannon(p, q), answer)

    def test_jensen_shannon_uniform(self):
        self.assertEqual(shannon.jensen_shannon(P_UNI, Q_UNI), 0.0)

    def test_jensen_shannon_both_zero_uses_epsilon(self):
        result = shannon.jensen_shannon([0.0], [0.0])
        expected = 0.5 * (0.00001 * log(1.0) + 0.00001 * log(1.0))
        self.assertEqual(result, expected)

    # ── jensen_difference ────────────────────────────────────────────────

    def test_jensen_difference(self):
        answer = sum([
            (((0.389 * log(0.389)) + (0.35 * log(0.35))) / 2) - (
                        ((0.389 + 0.35) / 2) * log((0.389 + 0.35) / 2)),
            (((0.5 * log(0.5)) + (0.563 * log(0.563))) / 2) - (
                        ((0.5 + 0.563) / 2) * log((0.5 + 0.563) / 2)),
            (((0.25 * log(0.25)) + (0.4 * log(0.4))) / 2) - (
                        ((0.25 + 0.4) / 2) * log((0.25 + 0.4) / 2)),
            (((0.625 * log(0.625)) + (0.5 * log(0.5))) / 2) - (
                        ((0.625 + 0.5) / 2) * log((0.625 + 0.5) / 2)),
            (((0.00001 * log(0.00001)) + (0.00001 * log(0.00001))) / 2) - (
                        ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / 2)),
            (((0.00001 * log(0.00001)) + (0.00001 * log(0.00001))) / 2) - (
                        ((0.00001 + 0.00001) / 2) * log((0.00001 + 0.00001) / 2)),
            (((0.25 * log(0.25)) + (0.00001 * log(0.00001))) / 2) - (
                        ((0.25 + 0.00001) / 2) * log((0.25 + 0.00001) / 2)),
        ])
        self.assertEqual(shannon.jensen_difference(p=P_STD, q=Q_STD), answer)

    def test_jensen_difference_identical(self):
        self.assertEqual(shannon.jensen_difference(P_NZ, P_NZ), 0.0)

    def test_jensen_difference_single_identical(self):
        self.assertEqual(shannon.jensen_difference([0.5], [0.5]), 0.0)

    def test_jensen_difference_single_different(self):
        p_a, q_b = 0.3, 0.7
        answer = (((p_a * log(p_a)) + (q_b * log(q_b))) / 2) - (
                    ((p_a + q_b) / 2) * log((p_a + q_b) / 2))
        self.assertEqual(shannon.jensen_difference([p_a], [q_b]), answer)

    def test_jensen_difference_symmetric(self):
        self.assertEqual(shannon.jensen_difference(P_NZ, Q_NZ),
                         shannon.jensen_difference(Q_NZ, P_NZ))

    def test_jensen_difference_non_negative(self):
        self.assertGreaterEqual(shannon.jensen_difference(P_NZ, Q_NZ), 0.0)

    def test_jensen_difference_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sum([
            (((pi * log(pi)) + (qi * log(qi))) / 2) - (((pi + qi) / 2) * log((pi + qi) / 2))
            for pi, qi in zip(p, q)
        ])
        self.assertEqual(shannon.jensen_difference(p, q), answer)

    def test_jensen_difference_uniform(self):
        self.assertEqual(shannon.jensen_difference(P_UNI, Q_UNI), 0.0)

    def test_jensen_difference_both_zero_uses_epsilon(self):
        result = shannon.jensen_difference([0.0], [0.0])
        e = 0.00001
        expected = (((e * log(e)) + (e * log(e))) / 2) - (((e + e) / 2) * log((e + e) / 2))
        self.assertEqual(result, expected)

    def test_jensen_difference_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum([
            (((pi * log(pi)) + (qi * log(qi))) / 2) - (((pi + qi) / 2) * log((pi + qi) / 2))
            for pi, qi in zip(p, q)
        ])
        self.assertEqual(shannon.jensen_difference(p, q), answer)
