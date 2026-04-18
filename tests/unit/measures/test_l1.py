"""
Unit Test Cases for the L1 Family measures.
"""
import unittest
from math import log

from scikit_pierre.measures import l1

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestL1(unittest.TestCase):

    # ── sorensen ─────────────────────────────────────────────────────────

    def test_sorensen(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(l1.sorensen(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_sorensen_identical(self):
        self.assertEqual(l1.sorensen(P_NZ, P_NZ), 0.0)

    def test_sorensen_single_identical(self):
        self.assertEqual(l1.sorensen([0.5], [0.5]), 0.0)

    def test_sorensen_single_different(self):
        answer = abs(0.3 - 0.7) / (0.3 + 0.7)
        self.assertEqual(l1.sorensen([0.3], [0.7]), answer)

    def test_sorensen_symmetric(self):
        self.assertEqual(l1.sorensen(P_NZ, Q_NZ), l1.sorensen(Q_NZ, P_NZ))

    def test_sorensen_non_negative(self):
        self.assertGreaterEqual(l1.sorensen(P_STD, Q_STD), 0.0)

    def test_sorensen_at_most_one(self):
        self.assertLessEqual(l1.sorensen(P_NZ, Q_NZ), 1.0)

    def test_sorensen_all_zeros_uses_epsilon(self):
        result = l1.sorensen([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_sorensen_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        denom = 0.3 + 0.7 + 0.4 + 0.6
        self.assertAlmostEqual(l1.sorensen(p, q), num / denom, places=10)

    def test_sorensen_uniform(self):
        self.assertEqual(l1.sorensen(P_UNI, Q_UNI), 0.0)

    def test_sorensen_equals_czekanowski_divergence(self):
        from scikit_pierre.measures import intersection
        s = l1.sorensen(P_NZ, Q_NZ)
        c = intersection.czekanowski_divergence(P_NZ, Q_NZ)
        self.assertAlmostEqual(s, c, places=10)

    # ── gower ────────────────────────────────────────────────────────────

    def test_gower(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(l1.gower(p=P_STD, q=Q_STD), answer_num / 7)

    def test_gower_identical(self):
        self.assertEqual(l1.gower(P_NZ, P_NZ), 0.0)

    def test_gower_single_identical(self):
        self.assertEqual(l1.gower([0.5], [0.5]), 0.0)

    def test_gower_single_different(self):
        self.assertEqual(l1.gower([0.3], [0.7]), abs(0.3 - 0.7) / 1)

    def test_gower_all_zeros(self):
        self.assertEqual(l1.gower([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_gower_symmetric(self):
        self.assertEqual(l1.gower(P_NZ, Q_NZ), l1.gower(Q_NZ, P_NZ))

    def test_gower_non_negative(self):
        self.assertGreaterEqual(l1.gower(P_STD, Q_STD), 0.0)

    def test_gower_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (abs(0.3 - 0.4) + abs(0.7 - 0.6)) / 2
        self.assertEqual(l1.gower(p, q), answer)

    def test_gower_uniform(self):
        self.assertEqual(l1.gower(P_UNI, Q_UNI), 0.0)

    def test_gower_depends_on_length(self):
        p2 = [0.3, 0.7]
        q2 = [0.4, 0.6]
        p4 = p2 * 2
        q4 = q2 * 2
        # gower(p2, q2) should equal gower(p4, q4) since it divides by length
        self.assertEqual(l1.gower(p2, q2), l1.gower(p4, q4))

    def test_gower_large_values(self):
        p, q = [100.0], [150.0]
        self.assertEqual(l1.gower(p, q), 50.0)

    # ── soergel ───────────────────────────────────────────────────────────

    def test_soergel(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]),
                           max([0.625, 0.5]), max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(l1.soergel(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_soergel_identical(self):
        self.assertEqual(l1.soergel(P_NZ, P_NZ), 0.0)

    def test_soergel_single_identical(self):
        self.assertEqual(l1.soergel([0.5], [0.5]), 0.0)

    def test_soergel_single_different(self):
        answer = abs(0.3 - 0.7) / max(0.3, 0.7)
        self.assertEqual(l1.soergel([0.3], [0.7]), answer)

    def test_soergel_all_zeros_uses_epsilon(self):
        result = l1.soergel([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_soergel_symmetric(self):
        self.assertEqual(l1.soergel(P_NZ, Q_NZ), l1.soergel(Q_NZ, P_NZ))

    def test_soergel_non_negative(self):
        self.assertGreaterEqual(l1.soergel(P_STD, Q_STD), 0.0)

    def test_soergel_at_most_one(self):
        self.assertLessEqual(l1.soergel(P_NZ, Q_NZ), 1.0)

    def test_soergel_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        denom = max(0.3, 0.4) + max(0.7, 0.6)
        self.assertEqual(l1.soergel(p, q), num / denom)

    def test_soergel_uniform(self):
        self.assertEqual(l1.soergel(P_UNI, Q_UNI), 0.0)

    def test_soergel_orthogonal(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        answer = (1.0 + 1.0) / (1.0 + 1.0)
        self.assertEqual(l1.soergel(p, q), answer)

    # ── kulczynski_d ──────────────────────────────────────────────────────

    def test_kulczynski_d(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]),
                           min([0.625, 0.5]), min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(l1.kulczynski_d(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_kulczynski_d_identical_uses_epsilon(self):
        result = l1.kulczynski_d(P_NZ, P_NZ)
        self.assertEqual(result, 0.0 / 0.00001)

    def test_kulczynski_d_single_identical_uses_epsilon(self):
        result = l1.kulczynski_d([0.5], [0.5])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_kulczynski_d_single_different(self):
        answer = abs(0.3 - 0.7) / min(0.3, 0.7)
        self.assertEqual(l1.kulczynski_d([0.3], [0.7]), answer)

    def test_kulczynski_d_all_zeros_uses_epsilon(self):
        result = l1.kulczynski_d([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_kulczynski_d_symmetric(self):
        self.assertEqual(l1.kulczynski_d(P_NZ, Q_NZ), l1.kulczynski_d(Q_NZ, P_NZ))

    def test_kulczynski_d_non_negative(self):
        self.assertGreaterEqual(l1.kulczynski_d(P_STD, Q_STD), 0.0)

    def test_kulczynski_d_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        denom = min(0.3, 0.4) + min(0.7, 0.6)
        self.assertEqual(l1.kulczynski_d(p, q), num / denom)

    def test_kulczynski_d_large_values(self):
        p, q = [100.0], [150.0]
        answer = abs(100.0 - 150.0) / min(100.0, 150.0)
        self.assertEqual(l1.kulczynski_d(p, q), answer)

    def test_kulczynski_d_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        num = sum(abs(pi - qi) for pi, qi in zip(p, q))
        denom = sum(min(pi, qi) for pi, qi in zip(p, q))
        self.assertEqual(l1.kulczynski_d(p, q), num / denom)

    # ── canberra ─────────────────────────────────────────────────────────

    def test_canberra(self):
        answer = sum([
            abs(0.389 - 0.35) / (0.389 + 0.35), abs(0.5 - 0.563) / (0.5 + 0.563),
            abs(0.25 - 0.4) / (0.25 + 0.4), abs(0.625 - 0.5) / (0.625 + 0.5),
            abs(0.0 - 0.0) / 0.00001, abs(0.0 - 0.0) / 0.00001,
            abs(0.25 - 0.0) / (0.25 + 0.0)
        ])
        self.assertEqual(l1.canberra(p=P_STD, q=Q_STD), answer)

    def test_canberra_identical(self):
        self.assertEqual(l1.canberra(P_NZ, P_NZ), 0.0)

    def test_canberra_single_identical(self):
        self.assertEqual(l1.canberra([0.5], [0.5]), 0.0)

    def test_canberra_single_different(self):
        answer = abs(0.3 - 0.7) / (0.3 + 0.7)
        self.assertEqual(l1.canberra([0.3], [0.7]), answer)

    def test_canberra_all_zeros_uses_epsilon(self):
        result = l1.canberra([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_canberra_symmetric(self):
        self.assertEqual(l1.canberra(P_NZ, Q_NZ), l1.canberra(Q_NZ, P_NZ))

    def test_canberra_non_negative(self):
        self.assertGreaterEqual(l1.canberra(P_STD, Q_STD), 0.0)

    def test_canberra_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = abs(0.3 - 0.4) / (0.3 + 0.4) + abs(0.7 - 0.6) / (0.7 + 0.6)
        self.assertEqual(l1.canberra(p, q), answer)

    def test_canberra_uniform(self):
        self.assertEqual(l1.canberra(P_UNI, Q_UNI), 0.0)

    def test_canberra_at_most_n(self):
        # Each term is at most 1, so canberra <= len(p)
        result = l1.canberra(P_NZ, Q_NZ)
        self.assertLessEqual(result, len(P_NZ))

    def test_canberra_large_values(self):
        p, q = [100.0], [150.0]
        answer = abs(100.0 - 150.0) / (100.0 + 150.0)
        self.assertEqual(l1.canberra(p, q), answer)

    # ── lorentzian ────────────────────────────────────────────────────────

    def test_lorentzian(self):
        answer = sum([
            log(1 + abs(0.389 - 0.35)), log(1 + abs(0.5 - 0.563)),
            log(1 + abs(0.25 - 0.4)), log(1 + abs(0.625 - 0.5)),
            log(1 + abs(0.0 - 0.0)), log(1 + abs(0.0 - 0.0)),
            log(1 + abs(0.25 - 0.0))
        ])
        self.assertEqual(l1.lorentzian(p=P_STD, q=Q_STD), answer)

    def test_lorentzian_identical(self):
        self.assertEqual(l1.lorentzian(P_NZ, P_NZ), 0.0)

    def test_lorentzian_single_identical(self):
        self.assertEqual(l1.lorentzian([0.5], [0.5]), 0.0)

    def test_lorentzian_single_different(self):
        answer = log(1 + abs(0.3 - 0.7))
        self.assertEqual(l1.lorentzian([0.3], [0.7]), answer)

    def test_lorentzian_all_zeros(self):
        self.assertEqual(l1.lorentzian([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_lorentzian_symmetric(self):
        self.assertEqual(l1.lorentzian(P_NZ, Q_NZ), l1.lorentzian(Q_NZ, P_NZ))

    def test_lorentzian_non_negative(self):
        self.assertGreaterEqual(l1.lorentzian(P_STD, Q_STD), 0.0)

    def test_lorentzian_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = log(1 + abs(0.3 - 0.4)) + log(1 + abs(0.7 - 0.6))
        self.assertEqual(l1.lorentzian(p, q), answer)

    def test_lorentzian_uniform(self):
        self.assertEqual(l1.lorentzian(P_UNI, Q_UNI), 0.0)

    def test_lorentzian_large_values(self):
        p, q = [100.0], [150.0]
        answer = log(1 + abs(100.0 - 150.0))
        self.assertEqual(l1.lorentzian(p, q), answer)

    def test_lorentzian_p_zero_q_nonzero(self):
        answer = log(1 + abs(0.0 - 0.5))
        self.assertEqual(l1.lorentzian([0.0], [0.5]), answer)
