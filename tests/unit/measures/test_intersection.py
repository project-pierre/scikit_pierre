"""
Unit Test Cases for the Intersection Family measures.
"""
import unittest

from scikit_pierre.measures import intersection

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestIntersection(unittest.TestCase):

    # ── intersection_similarity ───────────────────────────────────────────

    def test_intersection_similarity(self):
        answer = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                      min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(intersection.intersection_similarity(p=P_STD, q=Q_STD), answer)

    def test_intersection_similarity_identical(self):
        result = intersection.intersection_similarity(P_NZ, P_NZ)
        expected = sum(P_NZ)
        self.assertEqual(result, expected)

    def test_intersection_similarity_all_zeros(self):
        self.assertEqual(intersection.intersection_similarity([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_intersection_similarity_single_identical(self):
        self.assertEqual(intersection.intersection_similarity([0.5], [0.5]), 0.5)

    def test_intersection_similarity_single_different(self):
        self.assertEqual(intersection.intersection_similarity([0.3], [0.7]), 0.3)

    def test_intersection_similarity_symmetric(self):
        self.assertEqual(intersection.intersection_similarity(P_NZ, Q_NZ),
                         intersection.intersection_similarity(Q_NZ, P_NZ))

    def test_intersection_similarity_two_elements(self):
        answer = min(0.3, 0.7) + min(0.7, 0.3)
        self.assertEqual(intersection.intersection_similarity([0.3, 0.7], [0.7, 0.3]), answer)

    def test_intersection_similarity_non_negative(self):
        self.assertGreaterEqual(intersection.intersection_similarity(P_STD, Q_STD), 0.0)

    def test_intersection_similarity_uniform(self):
        self.assertEqual(intersection.intersection_similarity(P_UNI, Q_UNI), 1.0)

    def test_intersection_similarity_p_zero_q_nonzero(self):
        self.assertEqual(intersection.intersection_similarity([0.0], [0.5]), 0.0)

    def test_intersection_similarity_at_most_min_sum(self):
        result = intersection.intersection_similarity(P_NZ, Q_NZ)
        self.assertLessEqual(result, sum(P_NZ))

    # ── intersection_divergence ───────────────────────────────────────────

    def test_intersection_divergence(self):
        answer = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                      abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(intersection.intersection_divergence(p=P_STD, q=Q_STD), (1 / 2) * answer)

    def test_intersection_divergence_identical(self):
        self.assertEqual(intersection.intersection_divergence(P_NZ, P_NZ), 0.0)

    def test_intersection_divergence_all_zeros(self):
        self.assertEqual(intersection.intersection_divergence([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_intersection_divergence_single_identical(self):
        self.assertEqual(intersection.intersection_divergence([0.5], [0.5]), 0.0)

    def test_intersection_divergence_single_different(self):
        self.assertEqual(intersection.intersection_divergence([0.3], [0.7]), 0.5 * abs(0.3 - 0.7))

    def test_intersection_divergence_symmetric(self):
        self.assertEqual(intersection.intersection_divergence(P_NZ, Q_NZ),
                         intersection.intersection_divergence(Q_NZ, P_NZ))

    def test_intersection_divergence_non_negative(self):
        self.assertGreaterEqual(intersection.intersection_divergence(P_STD, Q_STD), 0.0)

    def test_intersection_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 0.5 * (abs(0.3 - 0.4) + abs(0.7 - 0.6))
        self.assertEqual(intersection.intersection_divergence(p, q), answer)

    def test_intersection_divergence_uniform(self):
        self.assertEqual(intersection.intersection_divergence(P_UNI, Q_UNI), 0.0)

    def test_intersection_divergence_large_difference(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        answer = 0.5 * (1.0 + 1.0)
        self.assertEqual(intersection.intersection_divergence(p, q), answer)

    # ── wave_hedges ────────────────────────────────────────────────────────

    def test_wave_hedges(self):
        answer = sum([
            abs(0.389 - 0.35) / max([0.389, 0.35]), abs(0.5 - 0.563) / max([0.5, 0.563]),
            abs(0.25 - 0.4) / max([0.25, 0.4]), abs(0.625 - 0.5) / max([0.625, 0.5]),
            abs(0.0 - 0.0) / 0.00001, abs(0.0 - 0.0) / 0.00001,
            abs(0.25 - 0.0) / max([0.25, 0.0])
        ])
        self.assertEqual(intersection.wave_hedges(p=P_STD, q=Q_STD), answer)

    def test_wave_hedges_identical(self):
        self.assertEqual(intersection.wave_hedges(P_NZ, P_NZ), 0.0)

    def test_wave_hedges_all_zeros_uses_epsilon(self):
        result = intersection.wave_hedges([0.0, 0.0], [0.0, 0.0])
        expected = 2 * (abs(0.0 - 0.0) / 0.00001)
        self.assertEqual(result, expected)

    def test_wave_hedges_single_identical(self):
        self.assertEqual(intersection.wave_hedges([0.5], [0.5]), 0.0)

    def test_wave_hedges_single_different(self):
        answer = abs(0.3 - 0.7) / max(0.3, 0.7)
        self.assertEqual(intersection.wave_hedges([0.3], [0.7]), answer)

    def test_wave_hedges_symmetric(self):
        self.assertEqual(intersection.wave_hedges(P_NZ, Q_NZ),
                         intersection.wave_hedges(Q_NZ, P_NZ))

    def test_wave_hedges_non_negative(self):
        self.assertGreaterEqual(intersection.wave_hedges(P_STD, Q_STD), 0.0)

    def test_wave_hedges_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = abs(0.3 - 0.4) / max(0.3, 0.4) + abs(0.7 - 0.6) / max(0.7, 0.6)
        self.assertEqual(intersection.wave_hedges(p, q), answer)

    def test_wave_hedges_uniform(self):
        self.assertEqual(intersection.wave_hedges(P_UNI, Q_UNI), 0.0)

    def test_wave_hedges_max_value_is_1_per_element(self):
        result = intersection.wave_hedges([0.5, 0.5], [0.0, 0.0])
        expected = abs(0.5) / max(0.5, 0.0) + abs(0.5) / max(0.5, 0.0)
        self.assertEqual(result, expected)

    # ── czekanowski_similarity ─────────────────────────────────────────────

    def test_czekanowski_similarity(self):
        answer_num = 2 * sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]),
                               min([0.625, 0.5]), min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(intersection.czekanowski_similarity(p=P_STD, q=Q_STD),
                         answer_num / answer_deno)

    def test_czekanowski_similarity_identical(self):
        self.assertEqual(intersection.czekanowski_similarity(P_NZ, P_NZ), 1.0)

    def test_czekanowski_similarity_all_zeros_uses_epsilon(self):
        result = intersection.czekanowski_similarity([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_czekanowski_similarity_single_identical(self):
        self.assertEqual(intersection.czekanowski_similarity([0.5], [0.5]), 1.0)

    def test_czekanowski_similarity_single_different(self):
        answer = 2 * min(0.3, 0.7) / (0.3 + 0.7)
        self.assertEqual(intersection.czekanowski_similarity([0.3], [0.7]), answer)

    def test_czekanowski_similarity_symmetric(self):
        self.assertEqual(intersection.czekanowski_similarity(P_NZ, Q_NZ),
                         intersection.czekanowski_similarity(Q_NZ, P_NZ))

    def test_czekanowski_similarity_non_negative(self):
        self.assertGreaterEqual(intersection.czekanowski_similarity(P_NZ, Q_NZ), 0.0)

    def test_czekanowski_similarity_at_most_one(self):
        self.assertLessEqual(intersection.czekanowski_similarity(P_NZ, Q_NZ), 1.0)

    def test_czekanowski_similarity_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = 2 * (min(0.3, 0.4) + min(0.7, 0.6))
        denom = 0.3 + 0.7 + 0.4 + 0.6
        self.assertAlmostEqual(intersection.czekanowski_similarity(p, q), num / denom, places=10)

    def test_czekanowski_similarity_uniform(self):
        self.assertEqual(intersection.czekanowski_similarity(P_UNI, Q_UNI), 1.0)

    # ── czekanowski_divergence ─────────────────────────────────────────────

    def test_czekanowski_divergence(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(intersection.czekanowski_divergence(p=P_STD, q=Q_STD),
                         answer_num / answer_deno)

    def test_czekanowski_divergence_identical(self):
        self.assertEqual(intersection.czekanowski_divergence(P_NZ, P_NZ), 0.0)

    def test_czekanowski_divergence_single_identical(self):
        self.assertEqual(intersection.czekanowski_divergence([0.5], [0.5]), 0.0)

    def test_czekanowski_divergence_single_different(self):
        answer = abs(0.3 - 0.7) / (0.3 + 0.7)
        self.assertEqual(intersection.czekanowski_divergence([0.3], [0.7]), answer)

    def test_czekanowski_divergence_symmetric(self):
        self.assertEqual(intersection.czekanowski_divergence(P_NZ, Q_NZ),
                         intersection.czekanowski_divergence(Q_NZ, P_NZ))

    def test_czekanowski_divergence_non_negative(self):
        self.assertGreaterEqual(intersection.czekanowski_divergence(P_NZ, Q_NZ), 0.0)

    def test_czekanowski_divergence_at_most_one(self):
        self.assertLessEqual(intersection.czekanowski_divergence(P_NZ, Q_NZ), 1.0)

    def test_czekanowski_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        denom = 0.3 + 0.7 + 0.4 + 0.6
        self.assertAlmostEqual(intersection.czekanowski_divergence(p, q), num / denom, places=10)

    def test_czekanowski_divergence_plus_similarity_equals_one(self):
        sim = intersection.czekanowski_similarity(P_NZ, Q_NZ)
        div = intersection.czekanowski_divergence(P_NZ, Q_NZ)
        self.assertAlmostEqual(sim + div, 1.0, places=10)

    def test_czekanowski_divergence_uniform(self):
        self.assertEqual(intersection.czekanowski_divergence(P_UNI, Q_UNI), 0.0)

    def test_czekanowski_divergence_all_zeros_uses_epsilon(self):
        result = intersection.czekanowski_divergence([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    # ── motyka_similarity ─────────────────────────────────────────────────

    def test_motyka_similarity(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]),
                          min([0.625, 0.5]), min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(intersection.motyka_similarity(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_motyka_similarity_identical(self):
        result = intersection.motyka_similarity(P_NZ, P_NZ)
        self.assertEqual(result, 0.5)

    def test_motyka_similarity_single_identical(self):
        self.assertEqual(intersection.motyka_similarity([0.5], [0.5]), 0.5)

    def test_motyka_similarity_single_different(self):
        answer = min(0.3, 0.7) / (0.3 + 0.7)
        self.assertEqual(intersection.motyka_similarity([0.3], [0.7]), answer)

    def test_motyka_similarity_symmetric(self):
        self.assertEqual(intersection.motyka_similarity(P_NZ, Q_NZ),
                         intersection.motyka_similarity(Q_NZ, P_NZ))

    def test_motyka_similarity_non_negative(self):
        self.assertGreaterEqual(intersection.motyka_similarity(P_NZ, Q_NZ), 0.0)

    def test_motyka_similarity_at_most_half(self):
        result = intersection.motyka_similarity(P_NZ, Q_NZ)
        self.assertLessEqual(result, 0.5)

    def test_motyka_similarity_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = min(0.3, 0.4) + min(0.7, 0.6)
        denom = 0.3 + 0.7 + 0.4 + 0.6
        self.assertAlmostEqual(intersection.motyka_similarity(p, q), num / denom, places=10)

    def test_motyka_similarity_uniform(self):
        self.assertEqual(intersection.motyka_similarity(P_UNI, Q_UNI), 0.5)

    def test_motyka_similarity_all_zeros_uses_epsilon(self):
        result = intersection.motyka_similarity([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_motyka_similarity_is_half_czekanowski(self):
        m = intersection.motyka_similarity(P_NZ, Q_NZ)
        c = intersection.czekanowski_similarity(P_NZ, Q_NZ)
        self.assertAlmostEqual(m, c / 2, places=10)

    # ── motyka_divergence ─────────────────────────────────────────────────

    def test_motyka_divergence(self):
        answer_num = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]),
                          max([0.625, 0.5]), max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(intersection.motyka_divergence(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_motyka_divergence_identical(self):
        result = intersection.motyka_divergence(P_NZ, P_NZ)
        self.assertEqual(result, 0.5)

    def test_motyka_divergence_single_identical(self):
        self.assertEqual(intersection.motyka_divergence([0.5], [0.5]), 0.5)

    def test_motyka_divergence_single_different(self):
        answer = max(0.3, 0.7) / (0.3 + 0.7)
        self.assertEqual(intersection.motyka_divergence([0.3], [0.7]), answer)

    def test_motyka_divergence_symmetric(self):
        self.assertEqual(intersection.motyka_divergence(P_NZ, Q_NZ),
                         intersection.motyka_divergence(Q_NZ, P_NZ))

    def test_motyka_divergence_at_least_half(self):
        result = intersection.motyka_divergence(P_NZ, Q_NZ)
        self.assertGreaterEqual(result, 0.5)

    def test_motyka_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = max(0.3, 0.4) + max(0.7, 0.6)
        denom = 0.3 + 0.7 + 0.4 + 0.6
        self.assertAlmostEqual(intersection.motyka_divergence(p, q), num / denom, places=10)

    def test_motyka_divergence_plus_similarity_equals_one(self):
        sim = intersection.motyka_similarity(P_NZ, Q_NZ)
        div = intersection.motyka_divergence(P_NZ, Q_NZ)
        self.assertAlmostEqual(sim + div, 1.0, places=10)

    def test_motyka_divergence_uniform(self):
        self.assertEqual(intersection.motyka_divergence(P_UNI, Q_UNI), 0.5)

    def test_motyka_divergence_all_zeros_uses_epsilon(self):
        result = intersection.motyka_divergence([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    # ── kulczynski_s ──────────────────────────────────────────────────────

    def test_kulczynski_s(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]),
                          min([0.625, 0.5]), min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                           abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(intersection.kulczynski_s(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_kulczynski_s_identical_uses_epsilon(self):
        result = intersection.kulczynski_s(P_NZ, P_NZ)
        num = sum(P_NZ)
        self.assertEqual(result, num / 0.00001)

    def test_kulczynski_s_single_identical_uses_epsilon(self):
        result = intersection.kulczynski_s([0.5], [0.5])
        self.assertEqual(result, 0.5 / 0.00001)

    def test_kulczynski_s_single_different(self):
        answer = min(0.3, 0.7) / abs(0.3 - 0.7)
        self.assertEqual(intersection.kulczynski_s([0.3], [0.7]), answer)

    def test_kulczynski_s_symmetric(self):
        self.assertEqual(intersection.kulczynski_s(P_NZ, Q_NZ),
                         intersection.kulczynski_s(Q_NZ, P_NZ))

    def test_kulczynski_s_non_negative(self):
        self.assertGreaterEqual(intersection.kulczynski_s(P_STD, Q_STD), 0.0)

    def test_kulczynski_s_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = min(0.3, 0.4) + min(0.7, 0.6)
        denom = abs(0.3 - 0.4) + abs(0.7 - 0.6)
        self.assertEqual(intersection.kulczynski_s(p, q), num / denom)

    def test_kulczynski_s_all_zeros_uses_epsilon(self):
        result = intersection.kulczynski_s([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_kulczynski_s_large_ratio(self):
        p, q = [0.5, 0.5], [0.5001, 0.4999]
        result = intersection.kulczynski_s(p, q)
        self.assertGreater(result, 0.0)

    def test_kulczynski_s_large_difference(self):
        p, q = [0.1, 0.9], [0.9, 0.1]
        num = min(0.1, 0.9) + min(0.9, 0.1)
        denom = abs(0.1 - 0.9) + abs(0.9 - 0.1)
        self.assertEqual(intersection.kulczynski_s(p, q), num / denom)

    # ── ruzicka ───────────────────────────────────────────────────────────

    def test_ruzicka(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]),
                          min([0.625, 0.5]), min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]),
                           max([0.625, 0.5]), max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(intersection.ruzicka(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_ruzicka_identical(self):
        self.assertEqual(intersection.ruzicka(P_NZ, P_NZ), 1.0)

    def test_ruzicka_single_identical(self):
        self.assertEqual(intersection.ruzicka([0.5], [0.5]), 1.0)

    def test_ruzicka_single_different(self):
        answer = min(0.3, 0.7) / max(0.3, 0.7)
        self.assertEqual(intersection.ruzicka([0.3], [0.7]), answer)

    def test_ruzicka_symmetric(self):
        self.assertEqual(intersection.ruzicka(P_NZ, Q_NZ),
                         intersection.ruzicka(Q_NZ, P_NZ))

    def test_ruzicka_non_negative(self):
        self.assertGreaterEqual(intersection.ruzicka(P_NZ, Q_NZ), 0.0)

    def test_ruzicka_at_most_one(self):
        self.assertLessEqual(intersection.ruzicka(P_NZ, Q_NZ), 1.0)

    def test_ruzicka_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = min(0.3, 0.4) + min(0.7, 0.6)
        denom = max(0.3, 0.4) + max(0.7, 0.6)
        self.assertEqual(intersection.ruzicka(p, q), num / denom)

    def test_ruzicka_all_zeros_uses_epsilon(self):
        result = intersection.ruzicka([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_ruzicka_uniform(self):
        self.assertEqual(intersection.ruzicka(P_UNI, Q_UNI), 1.0)

    def test_ruzicka_orthogonal(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        result = intersection.ruzicka(p, q)
        self.assertEqual(result, 0.0 / 1.0)

    # ── tanimoto ──────────────────────────────────────────────────────────

    def test_tanimoto(self):
        answer_num = sum([
            max([0.389, 0.35]) - min([0.389, 0.35]), max([0.5, 0.563]) - min([0.5, 0.563]),
            max([0.25, 0.4]) - min([0.25, 0.4]), max([0.625, 0.5]) - min([0.625, 0.5]),
            max([0.0, 0.0]) - min([0.0, 0.0]), max([0.0, 0.0]) - min([0.0, 0.0]),
            max([0.25, 0.0]) - min([0.25, 0.0])
        ])
        answer_deno = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]),
                           max([0.625, 0.5]), max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(intersection.tanimoto(p=P_STD, q=Q_STD), answer_num / answer_deno)

    def test_tanimoto_identical(self):
        self.assertEqual(intersection.tanimoto(P_NZ, P_NZ), 0.0)

    def test_tanimoto_single_identical(self):
        self.assertEqual(intersection.tanimoto([0.5], [0.5]), 0.0)

    def test_tanimoto_single_different(self):
        answer = (max(0.3, 0.7) - min(0.3, 0.7)) / max(0.3, 0.7)
        self.assertEqual(intersection.tanimoto([0.3], [0.7]), answer)

    def test_tanimoto_symmetric(self):
        self.assertEqual(intersection.tanimoto(P_NZ, Q_NZ),
                         intersection.tanimoto(Q_NZ, P_NZ))

    def test_tanimoto_non_negative(self):
        self.assertGreaterEqual(intersection.tanimoto(P_NZ, Q_NZ), 0.0)

    def test_tanimoto_at_most_one(self):
        self.assertLessEqual(intersection.tanimoto(P_NZ, Q_NZ), 1.0)

    def test_tanimoto_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = (max(0.3, 0.4) - min(0.3, 0.4)) + (max(0.7, 0.6) - min(0.7, 0.6))
        denom = max(0.3, 0.4) + max(0.7, 0.6)
        self.assertEqual(intersection.tanimoto(p, q), num / denom)

    def test_tanimoto_plus_ruzicka_equals_one(self):
        t = intersection.tanimoto(P_NZ, Q_NZ)
        r = intersection.ruzicka(P_NZ, Q_NZ)
        self.assertAlmostEqual(t + r, 1.0, places=10)

    def test_tanimoto_uniform(self):
        self.assertEqual(intersection.tanimoto(P_UNI, Q_UNI), 0.0)

    def test_tanimoto_all_zeros_uses_epsilon(self):
        result = intersection.tanimoto([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)
