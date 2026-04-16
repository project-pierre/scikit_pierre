"""
Unit Test Cases for the Inner Product Family measures.
"""
import unittest
from math import sqrt

from scikit_pierre.measures import inner_product

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestInnerProduct(unittest.TestCase):

    # ── inner_product ─────────────────────────────────────────────────────

    def test_inner_product(self):
        answer = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5,
                      0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        self.assertEqual(inner_product.inner_product(p=P_STD, q=Q_STD), answer)

    def test_inner_product_identical(self):
        result = inner_product.inner_product(P_NZ, P_NZ)
        expected = sum(v ** 2 for v in P_NZ)
        self.assertEqual(result, expected)

    def test_inner_product_all_zeros(self):
        self.assertEqual(inner_product.inner_product([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_inner_product_single_identical(self):
        self.assertEqual(inner_product.inner_product([0.5], [0.5]), 0.25)

    def test_inner_product_single_different(self):
        self.assertEqual(inner_product.inner_product([0.3], [0.7]), 0.3 * 0.7)

    def test_inner_product_symmetric(self):
        self.assertEqual(inner_product.inner_product(P_NZ, Q_NZ),
                         inner_product.inner_product(Q_NZ, P_NZ))

    def test_inner_product_non_negative_for_non_negative_inputs(self):
        self.assertGreaterEqual(inner_product.inner_product(P_STD, Q_STD), 0.0)

    def test_inner_product_orthogonal(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        self.assertEqual(inner_product.inner_product(p, q), 0.0)

    def test_inner_product_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 0.3 * 0.4 + 0.7 * 0.6
        self.assertEqual(inner_product.inner_product(p, q), answer)

    def test_inner_product_uniform(self):
        expected = sum(0.25 * 0.25 for _ in range(4))
        self.assertEqual(inner_product.inner_product(P_UNI, Q_UNI), expected)

    def test_inner_product_large_values(self):
        answer = 100.0 * 150.0 + 200.0 * 250.0
        self.assertEqual(inner_product.inner_product([100.0, 200.0], [150.0, 250.0]), answer)

    # ── harmonic_mean ─────────────────────────────────────────────────────

    def test_harmonic_mean(self):
        answer = 2 * sum([
            (0.389 * 0.35) / (0.389 + 0.35), (0.5 * 0.563) / (0.5 + 0.563),
            (0.25 * 0.4) / (0.25 + 0.4), (0.625 * 0.5) / (0.625 + 0.5),
            (0.0 * 0.0) / 0.00001, (0.0 * 0.0) / 0.00001,
            (0.25 * 0.0) / (0.25 + 0.0)
        ])
        self.assertEqual(inner_product.harmonic_mean(p=P_STD, q=Q_STD), answer)

    def test_harmonic_mean_identical(self):
        result = inner_product.harmonic_mean(P_NZ, P_NZ)
        expected = 2 * sum((v * v) / (v + v) for v in P_NZ)
        self.assertEqual(result, expected)

    def test_harmonic_mean_single_identical(self):
        answer = 2 * (0.5 * 0.5) / (0.5 + 0.5)
        self.assertEqual(inner_product.harmonic_mean([0.5], [0.5]), answer)

    def test_harmonic_mean_single_different(self):
        answer = 2 * (0.3 * 0.7) / (0.3 + 0.7)
        self.assertEqual(inner_product.harmonic_mean([0.3], [0.7]), answer)

    def test_harmonic_mean_both_zero_uses_epsilon(self):
        answer = 2 * (0.0 * 0.0) / 0.00001
        self.assertEqual(inner_product.harmonic_mean([0.0], [0.0]), answer)

    def test_harmonic_mean_symmetric(self):
        self.assertEqual(inner_product.harmonic_mean(P_NZ, Q_NZ),
                         inner_product.harmonic_mean(Q_NZ, P_NZ))

    def test_harmonic_mean_non_negative(self):
        self.assertGreaterEqual(inner_product.harmonic_mean(P_STD, Q_STD), 0.0)

    def test_harmonic_mean_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 2 * ((0.3 * 0.4) / (0.3 + 0.4) + (0.7 * 0.6) / (0.7 + 0.6))
        self.assertEqual(inner_product.harmonic_mean(p, q), answer)

    def test_harmonic_mean_uniform(self):
        expected = 2 * sum((0.25 * 0.25) / (0.25 + 0.25) for _ in range(4))
        self.assertEqual(inner_product.harmonic_mean(P_UNI, Q_UNI), expected)

    def test_harmonic_mean_p_zero_q_nonzero(self):
        answer = 2 * (0.0 * 0.5) / (0.0 + 0.5)
        self.assertEqual(inner_product.harmonic_mean([0.0], [0.5]), answer)

    def test_harmonic_mean_all_zeros(self):
        self.assertEqual(inner_product.harmonic_mean([0.0, 0.0], [0.0, 0.0]), 0.0)

    # ── cosine ────────────────────────────────────────────────────────────

    def test_cosine(self):
        answer_num = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5,
                          0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denom = sqrt(sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2,
                                 0.0 ** 2, 0.0 ** 2, 0.25 ** 2])) * \
                       sqrt(sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2,
                                 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.cosine(p=P_STD, q=Q_STD), answer_num / answer_denom)

    def test_cosine_identical(self):
        result = inner_product.cosine(P_NZ, P_NZ)
        num = sum(v ** 2 for v in P_NZ)
        denom = sqrt(sum(v ** 2 for v in P_NZ)) ** 2
        self.assertEqual(result, num / denom)

    def test_cosine_single_identical(self):
        self.assertEqual(inner_product.cosine([0.5], [0.5]), 1.0)

    def test_cosine_single_different(self):
        answer = (0.3 * 0.7) / (sqrt(0.3 ** 2) * sqrt(0.7 ** 2))
        self.assertEqual(inner_product.cosine([0.3], [0.7]), answer)

    def test_cosine_symmetric(self):
        self.assertEqual(inner_product.cosine(P_NZ, Q_NZ),
                         inner_product.cosine(Q_NZ, P_NZ))

    def test_cosine_orthogonal_vectors(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        self.assertEqual(inner_product.cosine(p, q), 0.0)

    def test_cosine_value_between_0_and_1(self):
        result = inner_product.cosine(P_NZ, Q_NZ)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_cosine_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = 0.3 * 0.4 + 0.7 * 0.6
        denom = sqrt(0.3 ** 2 + 0.7 ** 2) * sqrt(0.4 ** 2 + 0.6 ** 2)
        self.assertEqual(inner_product.cosine(p, q), num / denom)

    def test_cosine_uniform_distributions(self):
        result = inner_product.cosine(P_UNI, Q_UNI)
        self.assertEqual(result, 1.0)

    def test_cosine_all_zeros_uses_epsilon(self):
        result = inner_product.cosine([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_cosine_parallel_vectors(self):
        p = [1.0, 2.0, 3.0]
        q = [2.0, 4.0, 6.0]
        result = inner_product.cosine(p, q)
        self.assertAlmostEqual(result, 1.0, places=10)

    # ── kumar_hassebrook ──────────────────────────────────────────────────

    def test_kumar_hassebrook(self):
        answer_num = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5,
                          0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denom = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                        sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2])) - answer_num
        self.assertEqual(inner_product.kumar_hassebrook(p=P_STD, q=Q_STD), answer_num / answer_denom)

    def test_kumar_hassebrook_identical(self):
        result = inner_product.kumar_hassebrook(P_NZ, P_NZ)
        num = sum(v ** 2 for v in P_NZ)
        denom = 2 * sum(v ** 2 for v in P_NZ) - num
        self.assertEqual(result, num / denom)

    def test_kumar_hassebrook_single_identical(self):
        result = inner_product.kumar_hassebrook([0.5], [0.5])
        self.assertEqual(result, 0.25 / (0.25 + 0.25 - 0.25))

    def test_kumar_hassebrook_single_different(self):
        num = 0.3 * 0.7
        denom = 0.3 ** 2 + 0.7 ** 2 - num
        self.assertEqual(inner_product.kumar_hassebrook([0.3], [0.7]), num / denom)

    def test_kumar_hassebrook_symmetric(self):
        self.assertEqual(inner_product.kumar_hassebrook(P_NZ, Q_NZ),
                         inner_product.kumar_hassebrook(Q_NZ, P_NZ))

    def test_kumar_hassebrook_non_negative(self):
        self.assertGreaterEqual(inner_product.kumar_hassebrook(P_NZ, Q_NZ), 0.0)

    def test_kumar_hassebrook_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = 0.3 * 0.4 + 0.7 * 0.6
        denom = (0.3 ** 2 + 0.7 ** 2) + (0.4 ** 2 + 0.6 ** 2) - num
        self.assertEqual(inner_product.kumar_hassebrook(p, q), num / denom)

    def test_kumar_hassebrook_uniform(self):
        num = sum(0.25 * 0.25 for _ in range(4))
        denom = sum(0.25 ** 2 for _ in range(4)) + sum(0.25 ** 2 for _ in range(4)) - num
        self.assertEqual(inner_product.kumar_hassebrook(P_UNI, Q_UNI), num / denom)

    def test_kumar_hassebrook_value_at_most_one(self):
        result = inner_product.kumar_hassebrook(P_NZ, Q_NZ)
        self.assertLessEqual(result, 1.0)

    def test_kumar_hassebrook_orthogonal(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        result = inner_product.kumar_hassebrook(p, q)
        self.assertEqual(result, 0.0 / (1.0 + 1.0 - 0.0))

    def test_kumar_hassebrook_large_values(self):
        p, q = [10.0], [10.0]
        result = inner_product.kumar_hassebrook(p, q)
        self.assertEqual(result, 100.0 / (100.0 + 100.0 - 100.0))

    # ── jaccard ───────────────────────────────────────────────────────────

    def test_jaccard(self):
        answer_num = sum([(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2,
                          (0.625 - 0.5) ** 2, (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        answer_denom = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                        sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2])) - \
                       sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5,
                            0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        self.assertEqual(inner_product.jaccard(p=P_STD, q=Q_STD), answer_num / answer_denom)

    def test_jaccard_identical(self):
        self.assertEqual(inner_product.jaccard(P_NZ, P_NZ), 0.0)

    def test_jaccard_single_identical(self):
        self.assertEqual(inner_product.jaccard([0.5], [0.5]), 0.0)

    def test_jaccard_single_different(self):
        num = (0.3 - 0.7) ** 2
        denom = 0.3 ** 2 + 0.7 ** 2 - 0.3 * 0.7
        self.assertEqual(inner_product.jaccard([0.3], [0.7]), num / denom)

    def test_jaccard_symmetric(self):
        self.assertEqual(inner_product.jaccard(P_NZ, Q_NZ),
                         inner_product.jaccard(Q_NZ, P_NZ))

    def test_jaccard_non_negative(self):
        self.assertGreaterEqual(inner_product.jaccard(P_STD, Q_STD), 0.0)

    def test_jaccard_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = (0.3 - 0.4) ** 2 + (0.7 - 0.6) ** 2
        denom = (0.3 ** 2 + 0.7 ** 2) + (0.4 ** 2 + 0.6 ** 2) - (0.3 * 0.4 + 0.7 * 0.6)
        self.assertEqual(inner_product.jaccard(p, q), num / denom)

    def test_jaccard_uniform(self):
        self.assertEqual(inner_product.jaccard(P_UNI, Q_UNI), 0.0)

    def test_jaccard_at_most_one(self):
        result = inner_product.jaccard(P_NZ, Q_NZ)
        self.assertLessEqual(result, 1.0)

    def test_jaccard_all_zeros_uses_epsilon(self):
        result = inner_product.jaccard([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_jaccard_equals_one_minus_kumar_hassebrook_for_nz(self):
        jac = inner_product.jaccard(P_NZ, Q_NZ)
        kh = inner_product.kumar_hassebrook(P_NZ, Q_NZ)
        self.assertAlmostEqual(jac + kh, 1.0, places=10)

    # ── dice_similarity ───────────────────────────────────────────────────

    def test_dice_similarity(self):
        answer_num = 2 * sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5,
                               0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denom = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                        sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.dice_similarity(p=P_STD, q=Q_STD), answer_num / answer_denom)

    def test_dice_similarity_identical(self):
        result = inner_product.dice_similarity(P_NZ, P_NZ)
        self.assertEqual(result, 1.0)

    def test_dice_similarity_single_identical(self):
        self.assertEqual(inner_product.dice_similarity([0.5], [0.5]), 1.0)

    def test_dice_similarity_single_different(self):
        num = 2 * 0.3 * 0.7
        denom = 0.3 ** 2 + 0.7 ** 2
        self.assertEqual(inner_product.dice_similarity([0.3], [0.7]), num / denom)

    def test_dice_similarity_symmetric(self):
        self.assertEqual(inner_product.dice_similarity(P_NZ, Q_NZ),
                         inner_product.dice_similarity(Q_NZ, P_NZ))

    def test_dice_similarity_non_negative(self):
        self.assertGreaterEqual(inner_product.dice_similarity(P_NZ, Q_NZ), 0.0)

    def test_dice_similarity_at_most_one(self):
        self.assertLessEqual(inner_product.dice_similarity(P_NZ, Q_NZ), 1.0)

    def test_dice_similarity_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = 2 * (0.3 * 0.4 + 0.7 * 0.6)
        denom = (0.3 ** 2 + 0.7 ** 2) + (0.4 ** 2 + 0.6 ** 2)
        self.assertEqual(inner_product.dice_similarity(p, q), num / denom)

    def test_dice_similarity_all_zeros_uses_epsilon(self):
        result = inner_product.dice_similarity([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_dice_similarity_uniform(self):
        expected = 2 * 4 * 0.0625 / (2 * 4 * 0.0625)
        self.assertEqual(inner_product.dice_similarity(P_UNI, Q_UNI), expected)

    def test_dice_similarity_orthogonal(self):
        p, q = [1.0, 0.0], [0.0, 1.0]
        self.assertEqual(inner_product.dice_similarity(p, q), 0.0)

    # ── dice_divergence ───────────────────────────────────────────────────

    def test_dice_divergence(self):
        answer_num = sum([(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2,
                          (0.625 - 0.5) ** 2, (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        answer_denom = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                        sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.dice_divergence(p=P_STD, q=Q_STD), answer_num / answer_denom)

    def test_dice_divergence_identical(self):
        self.assertEqual(inner_product.dice_divergence(P_NZ, P_NZ), 0.0)

    def test_dice_divergence_single_identical(self):
        self.assertEqual(inner_product.dice_divergence([0.5], [0.5]), 0.0)

    def test_dice_divergence_single_different(self):
        num = (0.3 - 0.7) ** 2
        denom = 0.3 ** 2 + 0.7 ** 2
        self.assertEqual(inner_product.dice_divergence([0.3], [0.7]), num / denom)

    def test_dice_divergence_symmetric(self):
        self.assertEqual(inner_product.dice_divergence(P_NZ, Q_NZ),
                         inner_product.dice_divergence(Q_NZ, P_NZ))

    def test_dice_divergence_non_negative(self):
        self.assertGreaterEqual(inner_product.dice_divergence(P_STD, Q_STD), 0.0)

    def test_dice_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        num = (0.3 - 0.4) ** 2 + (0.7 - 0.6) ** 2
        denom = (0.3 ** 2 + 0.7 ** 2) + (0.4 ** 2 + 0.6 ** 2)
        self.assertEqual(inner_product.dice_divergence(p, q), num / denom)

    def test_dice_divergence_all_zeros_uses_epsilon(self):
        result = inner_product.dice_divergence([0.0], [0.0])
        self.assertEqual(result, 0.0 / 0.00001)

    def test_dice_divergence_at_most_one(self):
        self.assertLessEqual(inner_product.dice_divergence(P_NZ, Q_NZ), 1.0)

    def test_dice_divergence_uniform(self):
        self.assertEqual(inner_product.dice_divergence(P_UNI, Q_UNI), 0.0)

    def test_dice_divergence_plus_similarity_equals_one(self):
        sim = inner_product.dice_similarity(P_NZ, Q_NZ)
        div = inner_product.dice_divergence(P_NZ, Q_NZ)
        self.assertAlmostEqual(sim + div, 1.0, places=10)
