"""
Unit Test Cases for the Minkowski Family measures.
"""
import unittest
from math import sqrt

from scikit_pierre.measures import minkowski

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestMinkowski(unittest.TestCase):

    # ── city_block ────────────────────────────────────────────────────────

    def test_city_block(self):
        answer = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4),
                      abs(0.625 - 0.5), abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(minkowski.city_block(p=P_STD, q=Q_STD), answer)

    def test_city_block_identical(self):
        self.assertEqual(minkowski.city_block(P_NZ, P_NZ), 0.0)

    def test_city_block_all_zeros(self):
        self.assertEqual(minkowski.city_block([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_city_block_single_identical(self):
        self.assertEqual(minkowski.city_block([0.5], [0.5]), 0.0)

    def test_city_block_single_different(self):
        self.assertEqual(minkowski.city_block([0.3], [0.7]), abs(0.3 - 0.7))

    def test_city_block_symmetric(self):
        self.assertEqual(minkowski.city_block(P_NZ, Q_NZ), minkowski.city_block(Q_NZ, P_NZ))

    def test_city_block_non_negative(self):
        self.assertGreaterEqual(minkowski.city_block(P_STD, Q_STD), 0.0)

    def test_city_block_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        self.assertEqual(minkowski.city_block(p, q), answer)

    def test_city_block_uniform(self):
        self.assertEqual(minkowski.city_block(P_UNI, Q_UNI), 0.0)

    def test_city_block_large_values(self):
        p, q = [100.0, 200.0], [150.0, 250.0]
        answer = abs(100.0 - 150.0) + abs(200.0 - 250.0)
        self.assertEqual(minkowski.city_block(p, q), answer)

    def test_city_block_equals_minkowski_d1(self):
        result_cb = minkowski.city_block(P_NZ, Q_NZ)
        result_mk = minkowski.minkowski(P_NZ, Q_NZ, d=1)
        self.assertAlmostEqual(result_cb, result_mk, places=10)

    # ── euclidean ────────────────────────────────────────────────────────

    def test_euclidean(self):
        answer = sqrt(sum([abs(0.389 - 0.35) ** 2, abs(0.5 - 0.563) ** 2, abs(0.25 - 0.4) ** 2,
                           abs(0.625 - 0.5) ** 2, abs(0.0 - 0.0) ** 2, abs(0.0 - 0.0) ** 2,
                           abs(0.25 - 0.0) ** 2]))
        self.assertEqual(minkowski.euclidean(p=P_STD, q=Q_STD), answer)

    def test_euclidean_identical(self):
        self.assertEqual(minkowski.euclidean(P_NZ, P_NZ), 0.0)

    def test_euclidean_all_zeros(self):
        self.assertEqual(minkowski.euclidean([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_euclidean_single_identical(self):
        self.assertEqual(minkowski.euclidean([0.5], [0.5]), 0.0)

    def test_euclidean_single_different(self):
        self.assertEqual(minkowski.euclidean([0.0], [1.0]), 1.0)

    def test_euclidean_symmetric(self):
        self.assertEqual(minkowski.euclidean(P_NZ, Q_NZ), minkowski.euclidean(Q_NZ, P_NZ))

    def test_euclidean_non_negative(self):
        self.assertGreaterEqual(minkowski.euclidean(P_STD, Q_STD), 0.0)

    def test_euclidean_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sqrt((0.3 - 0.4) ** 2 + (0.7 - 0.6) ** 2)
        self.assertEqual(minkowski.euclidean(p, q), answer)

    def test_euclidean_uniform(self):
        self.assertEqual(minkowski.euclidean(P_UNI, Q_UNI), 0.0)

    def test_euclidean_3_4_5_triangle(self):
        self.assertEqual(minkowski.euclidean([0.0, 0.0], [3.0, 4.0]), 5.0)

    def test_euclidean_equals_minkowski_d2(self):
        result_eu = minkowski.euclidean(P_NZ, Q_NZ)
        result_mk = minkowski.minkowski(P_NZ, Q_NZ, d=2)
        self.assertAlmostEqual(result_eu, result_mk, places=10)

    # ── chebyshev ────────────────────────────────────────────────────────

    def test_cheb(self):
        answer = max([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4),
                      abs(0.625 - 0.5), abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(minkowski.chebyshev(p=P_STD, q=Q_STD), answer)

    def test_chebyshev_identical(self):
        self.assertEqual(minkowski.chebyshev(P_NZ, P_NZ), 0.0)

    def test_chebyshev_all_zeros(self):
        self.assertEqual(minkowski.chebyshev([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_chebyshev_single_identical(self):
        self.assertEqual(minkowski.chebyshev([0.5], [0.5]), 0.0)

    def test_chebyshev_single_different(self):
        self.assertEqual(minkowski.chebyshev([0.3], [0.7]), abs(0.3 - 0.7))

    def test_chebyshev_symmetric(self):
        self.assertEqual(minkowski.chebyshev(P_NZ, Q_NZ), minkowski.chebyshev(Q_NZ, P_NZ))

    def test_chebyshev_non_negative(self):
        self.assertGreaterEqual(minkowski.chebyshev(P_STD, Q_STD), 0.0)

    def test_chebyshev_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = max(abs(0.3 - 0.4), abs(0.7 - 0.6))
        self.assertEqual(minkowski.chebyshev(p, q), answer)

    def test_chebyshev_uniform(self):
        self.assertEqual(minkowski.chebyshev(P_UNI, Q_UNI), 0.0)

    def test_chebyshev_dominated_by_max_diff(self):
        p = [0.1, 0.9]
        q = [0.9, 0.1]
        self.assertEqual(minkowski.chebyshev(p, q), 0.8)

    def test_chebyshev_less_than_or_equal_city_block(self):
        result_cheb = minkowski.chebyshev(P_NZ, Q_NZ)
        result_cb = minkowski.city_block(P_NZ, Q_NZ)
        self.assertLessEqual(result_cheb, result_cb)

    # ── minkowski ─────────────────────────────────────────────────────────

    def test_minkowski(self):
        answer = sum([abs(0.389 - 0.35) ** 3, abs(0.5 - 0.563) ** 3, abs(0.25 - 0.4) ** 3,
                      abs(0.625 - 0.5) ** 3, abs(0.0 - 0.0) ** 3, abs(0.0 - 0.0) ** 3,
                      abs(0.25 - 0.0) ** 3]) ** (1 / 3)
        self.assertEqual(minkowski.minkowski(p=P_STD, q=Q_STD), answer)

    def test_minkowski_identical(self):
        self.assertEqual(minkowski.minkowski(P_NZ, P_NZ), 0.0)

    def test_minkowski_all_zeros(self):
        self.assertEqual(minkowski.minkowski([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_minkowski_single_identical(self):
        self.assertEqual(minkowski.minkowski([0.5], [0.5]), 0.0)

    def test_minkowski_single_different(self):
        self.assertEqual(minkowski.minkowski([0.0], [1.0]), 1.0)

    def test_minkowski_symmetric(self):
        self.assertEqual(minkowski.minkowski(P_NZ, Q_NZ), minkowski.minkowski(Q_NZ, P_NZ))

    def test_minkowski_non_negative(self):
        self.assertGreaterEqual(minkowski.minkowski(P_STD, Q_STD), 0.0)

    def test_minkowski_d_equals_1_is_city_block(self):
        result_mk = minkowski.minkowski(P_NZ, Q_NZ, d=1)
        result_cb = minkowski.city_block(P_NZ, Q_NZ)
        self.assertAlmostEqual(result_mk, result_cb, places=10)

    def test_minkowski_d_equals_2_is_euclidean(self):
        result_mk = minkowski.minkowski(P_NZ, Q_NZ, d=2)
        result_eu = minkowski.euclidean(P_NZ, Q_NZ)
        self.assertAlmostEqual(result_mk, result_eu, places=10)

    def test_minkowski_d3_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (abs(0.3 - 0.4) ** 3 + abs(0.7 - 0.6) ** 3) ** (1 / 3)
        self.assertEqual(minkowski.minkowski(p, q, d=3), answer)

    def test_minkowski_uniform(self):
        self.assertEqual(minkowski.minkowski(P_UNI, Q_UNI), 0.0)
