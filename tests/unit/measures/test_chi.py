"""
Unit Test Cases for the Chi Family measures.
"""
import unittest
from math import sqrt

from scikit_pierre.measures import chi

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestChi(unittest.TestCase):

    # ── squared_euclidean ──────────────────────────────────────────────────

    def test_squared_euclidean(self):
        answer = sum([(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2,
                      (0.625 - 0.5) ** 2, (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        self.assertEqual(chi.squared_euclidean(p=P_STD, q=Q_STD), answer)

    def test_squared_euclidean_identical_distributions(self):
        self.assertEqual(chi.squared_euclidean(P_NZ, P_NZ), 0.0)

    def test_squared_euclidean_all_zeros(self):
        self.assertEqual(chi.squared_euclidean([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_squared_euclidean_single_element_identical(self):
        self.assertEqual(chi.squared_euclidean([0.5], [0.5]), 0.0)

    def test_squared_euclidean_single_element_different(self):
        self.assertEqual(chi.squared_euclidean([1.0], [0.0]), 1.0)

    def test_squared_euclidean_symmetric(self):
        self.assertEqual(chi.squared_euclidean(P_NZ, Q_NZ),
                         chi.squared_euclidean(Q_NZ, P_NZ))

    def test_squared_euclidean_uniform_distributions(self):
        self.assertEqual(chi.squared_euclidean(P_UNI, Q_UNI), 0.0)

    def test_squared_euclidean_two_elements(self):
        answer = (0.3 - 0.7) ** 2 + (0.7 - 0.3) ** 2
        self.assertEqual(chi.squared_euclidean([0.3, 0.7], [0.7, 0.3]), answer)

    def test_squared_euclidean_non_negative(self):
        self.assertGreaterEqual(chi.squared_euclidean(P_STD, Q_STD), 0.0)

    def test_squared_euclidean_large_values(self):
        p, q = [100.0, 200.0], [150.0, 250.0]
        answer = (100.0 - 150.0) ** 2 + (200.0 - 250.0) ** 2
        self.assertEqual(chi.squared_euclidean(p, q), answer)

    def test_squared_euclidean_p_zero_q_nonzero(self):
        self.assertEqual(chi.squared_euclidean([0.0], [0.5]), 0.25)

    def test_squared_euclidean_q_zero_p_nonzero(self):
        self.assertEqual(chi.squared_euclidean([0.5], [0.0]), 0.25)

    # ── person_chi_square ─────────────────────────────────────────────────

    def test_person_chi_square(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / 0.35, ((0.5 - 0.563) ** 2) / 0.563,
            ((0.25 - 0.4) ** 2) / 0.4, ((0.625 - 0.5) ** 2) / 0.5,
            ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.25 - 0.0) ** 2) / 0.00001
        ])
        self.assertEqual(chi.person_chi_square(p=P_STD, q=Q_STD), answer)

    def test_person_chi_square_identical_distributions(self):
        self.assertEqual(chi.person_chi_square(P_NZ, P_NZ), 0.0)

    def test_person_chi_square_single_identical(self):
        self.assertEqual(chi.person_chi_square([0.5], [0.5]), 0.0)

    def test_person_chi_square_single_different(self):
        answer = (1.0 - 0.5) ** 2 / 0.5
        self.assertEqual(chi.person_chi_square([1.0], [0.5]), answer)

    def test_person_chi_square_q_zero_uses_epsilon(self):
        answer = (0.5 - 0.0) ** 2 / 0.00001
        self.assertEqual(chi.person_chi_square([0.5], [0.0]), answer)

    def test_person_chi_square_both_zero(self):
        self.assertEqual(chi.person_chi_square([0.0], [0.0]), 0.0)

    def test_person_chi_square_non_negative(self):
        self.assertGreaterEqual(chi.person_chi_square(P_STD, Q_STD), 0.0)

    def test_person_chi_square_asymmetric(self):
        p = [0.3, 0.7]
        q = [0.4, 0.6]
        self.assertNotEqual(chi.person_chi_square(p, q), chi.person_chi_square(q, p))

    def test_person_chi_square_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) ** 2 / 0.4 + (0.7 - 0.6) ** 2 / 0.6
        self.assertEqual(chi.person_chi_square(p, q), answer)

    def test_person_chi_square_uniform(self):
        self.assertEqual(chi.person_chi_square(P_UNI, Q_UNI), 0.0)

    def test_person_chi_square_large_values(self):
        answer = (100.0 - 150.0) ** 2 / 150.0
        self.assertEqual(chi.person_chi_square([100.0], [150.0]), answer)

    def test_person_chi_square_all_zeros(self):
        self.assertEqual(chi.person_chi_square([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    # ── neyman_square ────────────────────────────────────────────────────

    def test_neyman_square(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / 0.389, ((0.5 - 0.563) ** 2) / 0.5,
            ((0.25 - 0.4) ** 2) / 0.25, ((0.625 - 0.5) ** 2) / 0.625,
            ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.25 - 0.0) ** 2) / 0.25
        ])
        self.assertEqual(chi.neyman_square(p=P_STD, q=Q_STD), answer)

    def test_neyman_square_identical_distributions(self):
        self.assertEqual(chi.neyman_square(P_NZ, P_NZ), 0.0)

    def test_neyman_square_single_identical(self):
        self.assertEqual(chi.neyman_square([0.5], [0.5]), 0.0)

    def test_neyman_square_single_different(self):
        answer = (0.5 - 1.0) ** 2 / 0.5
        self.assertEqual(chi.neyman_square([0.5], [1.0]), answer)

    def test_neyman_square_p_zero_uses_epsilon(self):
        answer = (0.00001 - 0.5) ** 2 / 0.00001
        self.assertEqual(chi.neyman_square([0.0], [0.5]), answer)

    def test_neyman_square_both_zero(self):
        self.assertEqual(chi.neyman_square([0.0], [0.0]), 0.0)

    def test_neyman_square_non_negative(self):
        self.assertGreaterEqual(chi.neyman_square(P_STD, Q_STD), 0.0)

    def test_neyman_square_asymmetric(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        self.assertNotEqual(chi.neyman_square(p, q), chi.neyman_square(q, p))

    def test_neyman_square_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) ** 2 / 0.3 + (0.7 - 0.6) ** 2 / 0.7
        self.assertEqual(chi.neyman_square(p, q), answer)

    def test_neyman_square_uniform(self):
        self.assertEqual(chi.neyman_square(P_UNI, Q_UNI), 0.0)

    def test_neyman_square_all_zeros(self):
        self.assertEqual(chi.neyman_square([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_neyman_square_large_values(self):
        answer = (100.0 - 150.0) ** 2 / 100.0
        self.assertEqual(chi.neyman_square([100.0], [150.0]), answer)

    # ── squared_chi_square ───────────────────────────────────────────────

    def test_squared_chi_square(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / (0.389 + 0.35), ((0.5 - 0.563) ** 2) / (0.5 + 0.563),
            ((0.25 - 0.4) ** 2) / (0.25 + 0.4), ((0.625 - 0.5) ** 2) / (0.625 + 0.5),
            ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.25 - 0.0) ** 2) / (0.25 + 0.0)
        ])
        self.assertEqual(chi.squared_chi_square(p=P_STD, q=Q_STD), answer)

    def test_squared_chi_square_identical_distributions(self):
        self.assertEqual(chi.squared_chi_square(P_NZ, P_NZ), 0.0)

    def test_squared_chi_square_single_identical(self):
        self.assertEqual(chi.squared_chi_square([0.5], [0.5]), 0.0)

    def test_squared_chi_square_single_different(self):
        answer = (0.3 - 0.7) ** 2 / (0.3 + 0.7)
        self.assertEqual(chi.squared_chi_square([0.3], [0.7]), answer)

    def test_squared_chi_square_both_zero(self):
        self.assertEqual(chi.squared_chi_square([0.0], [0.0]), 0.0)

    def test_squared_chi_square_symmetric(self):
        self.assertEqual(chi.squared_chi_square(P_NZ, Q_NZ),
                         chi.squared_chi_square(Q_NZ, P_NZ))

    def test_squared_chi_square_non_negative(self):
        self.assertGreaterEqual(chi.squared_chi_square(P_STD, Q_STD), 0.0)

    def test_squared_chi_square_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) ** 2 / (0.3 + 0.4) + (0.7 - 0.6) ** 2 / (0.7 + 0.6)
        self.assertEqual(chi.squared_chi_square(p, q), answer)

    def test_squared_chi_square_uniform(self):
        self.assertEqual(chi.squared_chi_square(P_UNI, Q_UNI), 0.0)

    def test_squared_chi_square_large_values(self):
        answer = (100.0 - 150.0) ** 2 / (100.0 + 150.0)
        self.assertEqual(chi.squared_chi_square([100.0], [150.0]), answer)

    def test_squared_chi_square_all_zeros(self):
        self.assertEqual(chi.squared_chi_square([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_squared_chi_square_p_zero_q_nonzero(self):
        answer = (0.0 - 0.5) ** 2 / (0.0 + 0.5)
        self.assertEqual(chi.squared_chi_square([0.0], [0.5]), answer)

    # ── probabilistic_symmetric_chi_square ───────────────────────────────

    def test_probabilistic_symmetric_chi_square(self):
        answer = 2 * sum([
            ((0.389 - 0.35) ** 2) / (0.389 + 0.35), ((0.5 - 0.563) ** 2) / (0.5 + 0.563),
            ((0.25 - 0.4) ** 2) / (0.25 + 0.4), ((0.625 - 0.5) ** 2) / (0.625 + 0.5),
            ((0.0 - 0.0) ** 2) / 0.00001, ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.25 - 0.0) ** 2) / (0.25 + 0.0)
        ])
        self.assertEqual(chi.probabilistic_symmetric_chi_square(p=P_STD, q=Q_STD), answer)

    def test_probabilistic_symmetric_chi_square_identical(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square(P_NZ, P_NZ), 0.0)

    def test_probabilistic_symmetric_chi_square_single_identical(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square([0.5], [0.5]), 0.0)

    def test_probabilistic_symmetric_chi_square_single_different(self):
        answer = 2 * (0.3 - 0.7) ** 2 / (0.3 + 0.7)
        self.assertEqual(chi.probabilistic_symmetric_chi_square([0.3], [0.7]), answer)

    def test_probabilistic_symmetric_chi_square_equals_twice_squared_chi(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square(P_NZ, Q_NZ),
                         2 * chi.squared_chi_square(P_NZ, Q_NZ))

    def test_probabilistic_symmetric_chi_square_symmetric(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square(P_NZ, Q_NZ),
                         chi.probabilistic_symmetric_chi_square(Q_NZ, P_NZ))

    def test_probabilistic_symmetric_chi_square_non_negative(self):
        self.assertGreaterEqual(chi.probabilistic_symmetric_chi_square(P_STD, Q_STD), 0.0)

    def test_probabilistic_symmetric_chi_square_uniform(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square(P_UNI, Q_UNI), 0.0)

    def test_probabilistic_symmetric_chi_square_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 2 * ((0.3 - 0.4) ** 2 / (0.3 + 0.4) + (0.7 - 0.6) ** 2 / (0.7 + 0.6))
        self.assertEqual(chi.probabilistic_symmetric_chi_square(p, q), answer)

    def test_probabilistic_symmetric_chi_square_all_zeros(self):
        self.assertEqual(chi.probabilistic_symmetric_chi_square([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_probabilistic_symmetric_chi_square_large_values(self):
        answer = 2 * (100.0 - 150.0) ** 2 / (100.0 + 150.0)
        self.assertEqual(chi.probabilistic_symmetric_chi_square([100.0], [150.0]), answer)

    def test_probabilistic_symmetric_chi_square_reversed_symmetric(self):
        p, q = [0.1, 0.5, 0.4], [0.3, 0.3, 0.4]
        self.assertEqual(chi.probabilistic_symmetric_chi_square(p, q),
                         chi.probabilistic_symmetric_chi_square(q, p))

    # ── divergence ───────────────────────────────────────────────────────

    def test_divergence(self):
        answer = 2 * sum([
            ((0.389 - 0.35) ** 2) / (0.389 + 0.35) ** 2,
            ((0.5 - 0.563) ** 2) / (0.5 + 0.563) ** 2,
            ((0.25 - 0.4) ** 2) / (0.25 + 0.4) ** 2,
            ((0.625 - 0.5) ** 2) / (0.625 + 0.5) ** 2,
            ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.0 - 0.0) ** 2) / 0.00001,
            ((0.25 - 0.0) ** 2) / (0.25 + 0.0) ** 2
        ])
        self.assertEqual(chi.divergence(p=P_STD, q=Q_STD), answer)

    def test_divergence_identical(self):
        self.assertEqual(chi.divergence(P_NZ, P_NZ), 0.0)

    def test_divergence_single_identical(self):
        self.assertEqual(chi.divergence([0.5], [0.5]), 0.0)

    def test_divergence_single_different(self):
        answer = 2 * (0.3 - 0.7) ** 2 / (0.3 + 0.7) ** 2
        self.assertEqual(chi.divergence([0.3], [0.7]), answer)

    def test_divergence_symmetric(self):
        self.assertEqual(chi.divergence(P_NZ, Q_NZ),
                         chi.divergence(Q_NZ, P_NZ))

    def test_divergence_non_negative(self):
        self.assertGreaterEqual(chi.divergence(P_STD, Q_STD), 0.0)

    def test_divergence_uniform(self):
        self.assertEqual(chi.divergence(P_UNI, Q_UNI), 0.0)

    def test_divergence_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = 2 * ((0.3 - 0.4) ** 2 / (0.3 + 0.4) ** 2
                      + (0.7 - 0.6) ** 2 / (0.7 + 0.6) ** 2)
        self.assertEqual(chi.divergence(p, q), answer)

    def test_divergence_all_zeros(self):
        self.assertEqual(chi.divergence([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_divergence_large_values(self):
        answer = 2 * (100.0 - 150.0) ** 2 / (100.0 + 150.0) ** 2
        self.assertEqual(chi.divergence([100.0], [150.0]), answer)

    def test_divergence_reversed_symmetric(self):
        p, q = [0.2, 0.5, 0.3], [0.4, 0.3, 0.3]
        self.assertEqual(chi.divergence(p, q), chi.divergence(q, p))

    # ── clark ────────────────────────────────────────────────────────────

    def test_clark(self):
        answer = sqrt(sum([
            (abs(0.389 - 0.35) / (0.389 + 0.35)) ** 2,
            (abs(0.5 - 0.563) / (0.5 + 0.563)) ** 2,
            (abs(0.25 - 0.4) / (0.25 + 0.4)) ** 2,
            (abs(0.625 - 0.5) / (0.625 + 0.5)) ** 2,
            (0.0 - 0.0) / 0.00001,
            (0.0 - 0.0) / 0.00001,
            (abs(0.25 - 0.0) / (0.25 + 0.0)) ** 2
        ]))
        self.assertEqual(chi.clark(p=P_STD, q=Q_STD), answer)

    def test_clark_identical(self):
        self.assertEqual(chi.clark(P_NZ, P_NZ), 0.0)

    def test_clark_single_identical(self):
        self.assertEqual(chi.clark([0.5], [0.5]), 0.0)

    def test_clark_single_different(self):
        answer = sqrt((abs(0.3 - 0.7) / (0.3 + 0.7)) ** 2)
        self.assertEqual(chi.clark([0.3], [0.7]), answer)

    def test_clark_symmetric(self):
        self.assertEqual(chi.clark(P_NZ, Q_NZ),
                         chi.clark(Q_NZ, P_NZ))

    def test_clark_non_negative(self):
        self.assertGreaterEqual(chi.clark(P_STD, Q_STD), 0.0)

    def test_clark_uniform(self):
        self.assertEqual(chi.clark(P_UNI, Q_UNI), 0.0)

    def test_clark_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = sqrt((abs(0.3 - 0.4) / (0.3 + 0.4)) ** 2
                      + (abs(0.7 - 0.6) / (0.7 + 0.6)) ** 2)
        self.assertEqual(chi.clark(p, q), answer)

    def test_clark_large_values(self):
        answer = sqrt((abs(100.0 - 150.0) / (100.0 + 150.0)) ** 2)
        self.assertEqual(chi.clark([100.0], [150.0]), answer)

    def test_clark_all_zeros(self):
        self.assertEqual(chi.clark([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)

    def test_clark_reversed_symmetric(self):
        p, q = [0.1, 0.5, 0.4], [0.3, 0.3, 0.4]
        self.assertEqual(chi.clark(p, q), chi.clark(q, p))

    # ── additive_symmetric_chi_squared ───────────────────────────────────

    def test_additive_symmetric_chi_squared(self):
        answer = sum([
            (((0.389 - 0.35) ** 2) * (0.389 + 0.35)) / (0.389 * 0.35),
            (((0.5 - 0.563) ** 2) * (0.5 + 0.563)) / (0.5 * 0.563),
            (((0.25 - 0.4) ** 2) * (0.25 + 0.4)) / (0.25 * 0.4),
            (((0.625 - 0.5) ** 2) * (0.625 + 0.5)) / (0.625 * 0.5),
            (((0.0 - 0.0) ** 2) * (0.0 + 0.0)) / 0.00001,
            (((0.0 - 0.0) ** 2) * (0.0 + 0.0)) / 0.00001,
            (((0.25 - 0.0) ** 2) * (0.25 + 0.0)) / 0.00001
        ])
        self.assertEqual(chi.additive_symmetric_chi_squared(p=P_STD, q=Q_STD), answer)

    def test_additive_symmetric_chi_squared_identical(self):
        self.assertEqual(chi.additive_symmetric_chi_squared(P_NZ, P_NZ), 0.0)

    def test_additive_symmetric_chi_squared_single_identical(self):
        self.assertEqual(chi.additive_symmetric_chi_squared([0.5], [0.5]), 0.0)

    def test_additive_symmetric_chi_squared_single_different(self):
        answer = (((0.3 - 0.7) ** 2) * (0.3 + 0.7)) / (0.3 * 0.7)
        self.assertEqual(chi.additive_symmetric_chi_squared([0.3], [0.7]), answer)

    def test_additive_symmetric_chi_squared_symmetric(self):
        self.assertEqual(chi.additive_symmetric_chi_squared(P_NZ, Q_NZ),
                         chi.additive_symmetric_chi_squared(Q_NZ, P_NZ))

    def test_additive_symmetric_chi_squared_non_negative(self):
        self.assertGreaterEqual(chi.additive_symmetric_chi_squared(P_STD, Q_STD), 0.0)

    def test_additive_symmetric_chi_squared_uniform(self):
        self.assertEqual(chi.additive_symmetric_chi_squared(P_UNI, Q_UNI), 0.0)

    def test_additive_symmetric_chi_squared_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (((0.3 - 0.4) ** 2) * (0.3 + 0.4)) / (0.3 * 0.4) + \
                 (((0.7 - 0.6) ** 2) * (0.7 + 0.6)) / (0.7 * 0.6)
        self.assertEqual(chi.additive_symmetric_chi_squared(p, q), answer)

    def test_additive_symmetric_chi_squared_both_zero(self):
        answer = (((0.0 - 0.0) ** 2) * (0.0 + 0.0)) / 0.00001
        self.assertEqual(chi.additive_symmetric_chi_squared([0.0], [0.0]), answer)

    def test_additive_symmetric_chi_squared_large_values(self):
        answer = (((100.0 - 150.0) ** 2) * (100.0 + 150.0)) / (100.0 * 150.0)
        self.assertEqual(chi.additive_symmetric_chi_squared([100.0], [150.0]), answer)

    def test_additive_symmetric_chi_squared_p_zero_q_nonzero(self):
        answer = (((0.0 - 0.5) ** 2) * (0.0 + 0.5)) / 0.00001
        self.assertEqual(chi.additive_symmetric_chi_squared([0.0], [0.5]), answer)