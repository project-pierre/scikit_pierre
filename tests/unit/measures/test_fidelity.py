"""
Unit Test Cases for the Fidelity Family measures.
"""
import unittest
from math import sqrt, log

from scikit_pierre.measures import fidelity

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestFidelity(unittest.TestCase):

    # ── fidelity ─────────────────────────────────────────────────────────

    def test_fidelity(self):
        answer = sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4), sqrt(0.625 * 0.5),
                      sqrt(0.0 * 0.0), sqrt(0.0 * 0.0), sqrt(0.25 * 0.0)])
        self.assertEqual(fidelity.fidelity(p=P_STD, q=Q_STD), answer)

    def test_fidelity_identical_distributions(self):
        result = fidelity.fidelity(P_NZ, P_NZ)
        expected = sum(sqrt(p_i * p_i) for p_i in P_NZ)
        self.assertEqual(result, expected)

    def test_fidelity_all_zeros(self):
        self.assertEqual(fidelity.fidelity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_fidelity_single_identical(self):
        self.assertEqual(fidelity.fidelity([0.5], [0.5]), sqrt(0.25))

    def test_fidelity_single_different(self):
        self.assertEqual(fidelity.fidelity([0.25], [0.64]), sqrt(0.25 * 0.64))

    def test_fidelity_symmetric(self):
        self.assertEqual(fidelity.fidelity(P_NZ, Q_NZ), fidelity.fidelity(Q_NZ, P_NZ))

    def test_fidelity_non_negative(self):
        self.assertGreaterEqual(fidelity.fidelity(P_STD, Q_STD), 0.0)

    def test_fidelity_uniform_distributions(self):
        expected = sum(sqrt(0.25 * 0.25) for _ in range(4))
        self.assertEqual(fidelity.fidelity(P_UNI, Q_UNI), expected)

    def test_fidelity_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sqrt(0.3 * 0.4) + sqrt(0.7 * 0.6)
        self.assertEqual(fidelity.fidelity(p, q), answer)

    def test_fidelity_p_zero_q_nonzero(self):
        self.assertEqual(fidelity.fidelity([0.0], [0.5]), 0.0)

    def test_fidelity_one_is_zero(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        self.assertEqual(fidelity.fidelity(p, q), 0.0)

    # ── bhattacharyya ─────────────────────────────────────────────────────

    def test_bhattacharyya(self):
        answer = -log(sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4),
                           sqrt(0.625 * 0.5), sqrt(0.0 * 0.0), sqrt(0.0 * 0.0), sqrt(0.25 * 0.0)]))
        self.assertEqual(fidelity.bhattacharyya(p=P_STD, q=Q_STD), answer)

    def test_bhattacharyya_identical_distributions(self):
        result = fidelity.bhattacharyya(P_NZ, P_NZ)
        inner = sum(sqrt(v * v) for v in P_NZ)
        expected = -log(inner)
        self.assertEqual(result, expected)

    def test_bhattacharyya_single_identical(self):
        self.assertEqual(fidelity.bhattacharyya([0.5], [0.5]), -log(sqrt(0.5 * 0.5)))

    def test_bhattacharyya_single_different(self):
        self.assertEqual(fidelity.bhattacharyya([0.25], [0.64]), -log(sqrt(0.25 * 0.64)))

    def test_bhattacharyya_symmetric(self):
        self.assertEqual(fidelity.bhattacharyya(P_NZ, Q_NZ),
                         fidelity.bhattacharyya(Q_NZ, P_NZ))

    def test_bhattacharyya_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = -log(sqrt(0.3 * 0.4) + sqrt(0.7 * 0.6))
        self.assertEqual(fidelity.bhattacharyya(p, q), answer)

    def test_bhattacharyya_all_zeros_uses_epsilon(self):
        result = fidelity.bhattacharyya([0.0, 0.0], [0.0, 0.0])
        self.assertEqual(result, -log(0.00001))

    def test_bhattacharyya_uniform_identical(self):
        inner = sum(sqrt(0.25 * 0.25) for _ in range(4))
        expected = -log(inner)
        self.assertEqual(fidelity.bhattacharyya(P_UNI, Q_UNI), expected)

    def test_bhattacharyya_non_negative_for_close_distributions(self):
        p, q = [0.4, 0.6], [0.5, 0.5]
        result = fidelity.bhattacharyya(p, q)
        inner = sqrt(0.4 * 0.5) + sqrt(0.6 * 0.5)
        self.assertEqual(result, -log(inner))

    def test_bhattacharyya_p_zero_q_nonzero_uses_epsilon(self):
        result = fidelity.bhattacharyya([0.0], [0.5])
        expected = -log(0.00001)
        self.assertEqual(result, expected)

    def test_bhattacharyya_large_inner_product(self):
        p = [1.0, 0.0]
        q = [1.0, 0.0]
        result = fidelity.bhattacharyya(p, q)
        self.assertEqual(result, -log(1.0))

    # ── hellinger ─────────────────────────────────────────────────────────

    def test_hellinger(self):
        answer = sqrt(2 * sum([
            (sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2,
            (sqrt(0.25) - sqrt(0.4)) ** 2, (sqrt(0.625) - sqrt(0.5)) ** 2,
            (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
            (sqrt(0.25) - sqrt(0.0)) ** 2
        ]))
        self.assertEqual(fidelity.hellinger(p=P_STD, q=Q_STD), answer)

    def test_hellinger_identical_distributions(self):
        self.assertEqual(fidelity.hellinger(P_NZ, P_NZ), 0.0)

    def test_hellinger_all_zeros(self):
        self.assertEqual(fidelity.hellinger([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_hellinger_single_identical(self):
        self.assertEqual(fidelity.hellinger([0.5], [0.5]), 0.0)

    def test_hellinger_single_different(self):
        answer = sqrt(2 * (sqrt(0.3) - sqrt(0.7)) ** 2)
        self.assertEqual(fidelity.hellinger([0.3], [0.7]), answer)

    def test_hellinger_symmetric(self):
        self.assertEqual(fidelity.hellinger(P_NZ, Q_NZ),
                         fidelity.hellinger(Q_NZ, P_NZ))

    def test_hellinger_non_negative(self):
        self.assertGreaterEqual(fidelity.hellinger(P_STD, Q_STD), 0.0)

    def test_hellinger_uniform_distributions(self):
        self.assertEqual(fidelity.hellinger(P_UNI, Q_UNI), 0.0)

    def test_hellinger_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sqrt(2 * ((sqrt(0.3) - sqrt(0.4)) ** 2 + (sqrt(0.7) - sqrt(0.6)) ** 2))
        self.assertEqual(fidelity.hellinger(p, q), answer)

    def test_hellinger_p_zero_q_nonzero(self):
        answer = sqrt(2 * (sqrt(0.0) - sqrt(0.5)) ** 2)
        self.assertEqual(fidelity.hellinger([0.0], [0.5]), answer)

    def test_hellinger_max_distance(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        answer = sqrt(2 * ((sqrt(1.0) - sqrt(0.0)) ** 2 + (sqrt(0.0) - sqrt(1.0)) ** 2))
        self.assertEqual(fidelity.hellinger(p, q), answer)

    # ── matusita ──────────────────────────────────────────────────────────

    def test_matusita(self):
        answer = sqrt(sum([
            (sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2,
            (sqrt(0.25) - sqrt(0.4)) ** 2, (sqrt(0.625) - sqrt(0.5)) ** 2,
            (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
            (sqrt(0.25) - sqrt(0.0)) ** 2
        ]))
        self.assertEqual(fidelity.matusita(p=P_STD, q=Q_STD), answer)

    def test_matusita_identical_distributions(self):
        self.assertEqual(fidelity.matusita(P_NZ, P_NZ), 0.0)

    def test_matusita_all_zeros(self):
        self.assertEqual(fidelity.matusita([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_matusita_single_identical(self):
        self.assertEqual(fidelity.matusita([0.5], [0.5]), 0.0)

    def test_matusita_single_different(self):
        answer = sqrt((sqrt(0.3) - sqrt(0.7)) ** 2)
        self.assertEqual(fidelity.matusita([0.3], [0.7]), answer)

    def test_matusita_symmetric(self):
        self.assertEqual(fidelity.matusita(P_NZ, Q_NZ),
                         fidelity.matusita(Q_NZ, P_NZ))

    def test_matusita_non_negative(self):
        self.assertGreaterEqual(fidelity.matusita(P_STD, Q_STD), 0.0)

    def test_matusita_uniform_distributions(self):
        self.assertEqual(fidelity.matusita(P_UNI, Q_UNI), 0.0)

    def test_matusita_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sqrt((sqrt(0.3) - sqrt(0.4)) ** 2 + (sqrt(0.7) - sqrt(0.6)) ** 2)
        self.assertEqual(fidelity.matusita(p, q), answer)

    def test_matusita_hellinger_relation(self):
        # hellinger = sqrt(2) * matusita
        h = fidelity.hellinger(P_NZ, Q_NZ)
        m = fidelity.matusita(P_NZ, Q_NZ)
        self.assertAlmostEqual(h, sqrt(2) * m, places=10)

    def test_matusita_p_zero_q_nonzero(self):
        answer = sqrt((sqrt(0.0) - sqrt(0.5)) ** 2)
        self.assertEqual(fidelity.matusita([0.0], [0.5]), answer)

    # ── squared_chord_similarity ──────────────────────────────────────────

    def test_squared_chord_similarity(self):
        answer = 2 * sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4),
                          sqrt(0.625 * 0.5), sqrt(0.0 * 0.0), sqrt(0.0 * 0.0),
                          sqrt(0.25 * 0.0)]) - 1
        self.assertEqual(fidelity.squared_chord_similarity(p=P_STD, q=Q_STD), answer)

    def test_squared_chord_similarity_identical(self):
        result = fidelity.squared_chord_similarity(P_NZ, P_NZ)
        expected = 2 * sum(sqrt(v * v) for v in P_NZ) - 1
        self.assertEqual(result, expected)

    def test_squared_chord_similarity_all_zeros(self):
        self.assertEqual(fidelity.squared_chord_similarity([0.0, 0.0], [0.0, 0.0]), -1.0)

    def test_squared_chord_similarity_single_identical(self):
        answer = 2 * sqrt(0.5 * 0.5) - 1
        self.assertEqual(fidelity.squared_chord_similarity([0.5], [0.5]), answer)

    def test_squared_chord_similarity_single_different(self):
        answer = 2 * sqrt(0.3 * 0.7) - 1
        self.assertEqual(fidelity.squared_chord_similarity([0.3], [0.7]), answer)

    def test_squared_chord_similarity_symmetric(self):
        self.assertEqual(fidelity.squared_chord_similarity(P_NZ, Q_NZ),
                         fidelity.squared_chord_similarity(Q_NZ, P_NZ))

    def test_squared_chord_similarity_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 2 * (sqrt(0.3 * 0.4) + sqrt(0.7 * 0.6)) - 1
        self.assertEqual(fidelity.squared_chord_similarity(p, q), answer)

    def test_squared_chord_similarity_equals_twice_fidelity_minus_one(self):
        self.assertEqual(fidelity.squared_chord_similarity(P_NZ, Q_NZ),
                         2 * fidelity.fidelity(P_NZ, Q_NZ) - 1)

    def test_squared_chord_similarity_p_zero_q_nonzero(self):
        answer = 2 * sqrt(0.0 * 0.5) - 1
        self.assertEqual(fidelity.squared_chord_similarity([0.0], [0.5]), answer)

    def test_squared_chord_similarity_uniform(self):
        expected = 2 * sum(sqrt(0.25 * 0.25) for _ in range(4)) - 1
        self.assertEqual(fidelity.squared_chord_similarity(P_UNI, Q_UNI), expected)

    def test_squared_chord_similarity_one_is_zero(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        self.assertEqual(fidelity.squared_chord_similarity(p, q), -1.0)

    # ── squared_chord_divergence ──────────────────────────────────────────

    def test_squared_chord_divergence(self):
        answer = sum([
            (sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2,
            (sqrt(0.25) - sqrt(0.4)) ** 2, (sqrt(0.625) - sqrt(0.5)) ** 2,
            (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
            (sqrt(0.25) - sqrt(0.0)) ** 2
        ])
        self.assertEqual(fidelity.squared_chord_divergence(p=P_STD, q=Q_STD), answer)

    def test_squared_chord_divergence_identical(self):
        self.assertEqual(fidelity.squared_chord_divergence(P_NZ, P_NZ), 0.0)

    def test_squared_chord_divergence_all_zeros(self):
        self.assertEqual(fidelity.squared_chord_divergence([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_squared_chord_divergence_single_identical(self):
        self.assertEqual(fidelity.squared_chord_divergence([0.5], [0.5]), 0.0)

    def test_squared_chord_divergence_single_different(self):
        answer = (sqrt(0.3) - sqrt(0.7)) ** 2
        self.assertEqual(fidelity.squared_chord_divergence([0.3], [0.7]), answer)

    def test_squared_chord_divergence_symmetric(self):
        self.assertEqual(fidelity.squared_chord_divergence(P_NZ, Q_NZ),
                         fidelity.squared_chord_divergence(Q_NZ, P_NZ))

    def test_squared_chord_divergence_non_negative(self):
        self.assertGreaterEqual(fidelity.squared_chord_divergence(P_STD, Q_STD), 0.0)

    def test_squared_chord_divergence_equals_matusita_squared(self):
        m = fidelity.matusita(P_NZ, Q_NZ)
        d = fidelity.squared_chord_divergence(P_NZ, Q_NZ)
        self.assertAlmostEqual(d, m ** 2, places=10)

    def test_squared_chord_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (sqrt(0.3) - sqrt(0.4)) ** 2 + (sqrt(0.7) - sqrt(0.6)) ** 2
        self.assertEqual(fidelity.squared_chord_divergence(p, q), answer)

    def test_squared_chord_divergence_uniform(self):
        self.assertEqual(fidelity.squared_chord_divergence(P_UNI, Q_UNI), 0.0)

    def test_squared_chord_divergence_p_zero_q_nonzero(self):
        answer = (sqrt(0.0) - sqrt(0.5)) ** 2
        self.assertEqual(fidelity.squared_chord_divergence([0.0], [0.5]), answer)
