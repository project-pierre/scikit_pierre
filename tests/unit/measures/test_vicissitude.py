"""
Unit Test Cases for the Vicissitude Family measures.
"""
import unittest

from scikit_pierre.measures import vicissitude

P_STD = [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
Q_STD = [0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]
P_NZ = [0.1, 0.2, 0.3, 0.4]
Q_NZ = [0.4, 0.3, 0.2, 0.1]
P_UNI = [0.25, 0.25, 0.25, 0.25]
Q_UNI = [0.25, 0.25, 0.25, 0.25]


class TestVicissitude(unittest.TestCase):

    # ── vicis_wave_hedges ─────────────────────────────────────────────────

    def test_vicis_wave_hedges(self):
        answer = sum([
            abs(0.389 - 0.35) / min([0.389, 0.35]), abs(0.5 - 0.563) / min([0.5, 0.563]),
            abs(0.25 - 0.4) / min([0.25, 0.4]), abs(0.625 - 0.5) / min([0.625, 0.5]),
            abs(0.00001 - 0.00001) / 0.00001, abs(0.00001 - 0.00001) / 0.00001,
            abs(0.25 - 0.00001) / min([0.25, 0.00001])
        ])
        self.assertEqual(vicissitude.vicis_wave_hedges(p=P_STD, q=Q_STD), answer)

    def test_vicis_wave_hedges_identical(self):
        self.assertEqual(vicissitude.vicis_wave_hedges(P_NZ, P_NZ), 0.0)

    def test_vicis_wave_hedges_single_identical(self):
        self.assertEqual(vicissitude.vicis_wave_hedges([0.5], [0.5]), 0.0)

    def test_vicis_wave_hedges_single_different(self):
        answer = abs(0.3 - 0.7) / min(0.3, 0.7)
        self.assertEqual(vicissitude.vicis_wave_hedges([0.3], [0.7]), answer)

    def test_vicis_wave_hedges_both_zero_uses_epsilon(self):
        result = vicissitude.vicis_wave_hedges([0.0], [0.0])
        expected = abs(0.00001 - 0.00001) / 0.00001
        self.assertEqual(result, expected)

    def test_vicis_wave_hedges_symmetric(self):
        self.assertEqual(vicissitude.vicis_wave_hedges(P_NZ, Q_NZ),
                         vicissitude.vicis_wave_hedges(Q_NZ, P_NZ))

    def test_vicis_wave_hedges_non_negative(self):
        self.assertGreaterEqual(vicissitude.vicis_wave_hedges(P_STD, Q_STD), 0.0)

    def test_vicis_wave_hedges_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = abs(0.3 - 0.4) / min(0.3, 0.4) + abs(0.7 - 0.6) / min(0.7, 0.6)
        self.assertEqual(vicissitude.vicis_wave_hedges(p, q), answer)

    def test_vicis_wave_hedges_uniform(self):
        self.assertEqual(vicissitude.vicis_wave_hedges(P_UNI, Q_UNI), 0.0)

    def test_vicis_wave_hedges_p_zero_uses_epsilon(self):
        result = vicissitude.vicis_wave_hedges([0.0], [0.5])
        expected = abs(0.00001 - 0.5) / 0.00001
        self.assertEqual(result, expected)

    def test_vicis_wave_hedges_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum(abs(pi - qi) / min(pi, qi) for pi, qi in zip(p, q))
        self.assertEqual(vicissitude.vicis_wave_hedges(p, q), answer)

    # ── vicis_symmetric_chi_square ────────────────────────────────────────

    def test_vicis_symmetric_chi_square(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / (min([0.389, 0.35])) ** 2,
            ((0.5 - 0.563) ** 2) / (min([0.5, 0.563])) ** 2,
            ((0.25 - 0.4) ** 2) / (min([0.25, 0.4])) ** 2,
            ((0.625 - 0.5) ** 2) / (min([0.625, 0.5])) ** 2,
            ((0.00001 - 0.00001) ** 2) / (0.00001 ** 2),
            ((0.00001 - 0.00001) ** 2) / (0.00001 ** 2),
            ((0.25 - 0.00001) ** 2) / (min([0.25, 0.00001]) ** 2)
        ])
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(p=P_STD, q=Q_STD), answer)

    def test_vicis_symmetric_chi_square_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(P_NZ, P_NZ), 0.0)

    def test_vicis_symmetric_chi_square_single_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square([0.5], [0.5]), 0.0)

    def test_vicis_symmetric_chi_square_single_different(self):
        answer = (0.3 - 0.7) ** 2 / (min(0.3, 0.7) ** 2)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square([0.3], [0.7]), answer)

    def test_vicis_symmetric_chi_square_both_zero_uses_epsilon(self):
        result = vicissitude.vicis_symmetric_chi_square([0.0], [0.0])
        expected = (0.00001 - 0.00001) ** 2 / (0.00001 ** 2)
        self.assertEqual(result, expected)

    def test_vicis_symmetric_chi_square_symmetric(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(P_NZ, Q_NZ),
                         vicissitude.vicis_symmetric_chi_square(Q_NZ, P_NZ))

    def test_vicis_symmetric_chi_square_non_negative(self):
        self.assertGreaterEqual(vicissitude.vicis_symmetric_chi_square(P_STD, Q_STD), 0.0)

    def test_vicis_symmetric_chi_square_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = ((0.3 - 0.4) ** 2 / min(0.3, 0.4) ** 2 +
                  (0.7 - 0.6) ** 2 / min(0.7, 0.6) ** 2)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(p, q), answer)

    def test_vicis_symmetric_chi_square_uniform(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(P_UNI, Q_UNI), 0.0)

    def test_vicis_symmetric_chi_square_p_zero_uses_epsilon(self):
        result = vicissitude.vicis_symmetric_chi_square([0.0], [0.5])
        expected = (0.00001 - 0.5) ** 2 / (0.00001 ** 2)
        self.assertEqual(result, expected)

    def test_vicis_symmetric_chi_square_three_elements(self):
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.3, 0.3]
        answer = sum((pi - qi) ** 2 / min(pi, qi) ** 2 for pi, qi in zip(p, q))
        self.assertEqual(vicissitude.vicis_symmetric_chi_square(p, q), answer)

    # ── vicis_symmetric_chi_square_emanon3 ───────────────────────────────

    def test_vicis_symmetric_chi_square_emanon3(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / min([0.389, 0.35]), ((0.5 - 0.563) ** 2) / min([0.5, 0.563]),
            ((0.25 - 0.4) ** 2) / min([0.25, 0.4]), ((0.625 - 0.5) ** 2) / min([0.625, 0.5]),
            ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
            ((0.25 - 0.00001) ** 2) / min([0.25, 0.00001])
        ])
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3(p=P_STD, q=Q_STD), answer)

    def test_vicis_symmetric_chi_square_emanon3_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3(P_NZ, P_NZ), 0.0)

    def test_vicis_symmetric_chi_square_emanon3_single_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3([0.5], [0.5]), 0.0)

    def test_vicis_symmetric_chi_square_emanon3_single_different(self):
        answer = (0.3 - 0.7) ** 2 / min(0.3, 0.7)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3([0.3], [0.7]), answer)

    def test_vicis_symmetric_chi_square_emanon3_both_zero_uses_epsilon(self):
        result = vicissitude.vicis_symmetric_chi_square_emanon3([0.0], [0.0])
        expected = (0.00001 - 0.00001) ** 2 / 0.00001
        self.assertEqual(result, expected)

    def test_vicis_symmetric_chi_square_emanon3_symmetric(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3(P_NZ, Q_NZ),
                         vicissitude.vicis_symmetric_chi_square_emanon3(Q_NZ, P_NZ))

    def test_vicis_symmetric_chi_square_emanon3_non_negative(self):
        self.assertGreaterEqual(vicissitude.vicis_symmetric_chi_square_emanon3(P_STD, Q_STD), 0.0)

    def test_vicis_symmetric_chi_square_emanon3_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) ** 2 / min(0.3, 0.4) + (0.7 - 0.6) ** 2 / min(0.7, 0.6)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3(p, q), answer)

    def test_vicis_symmetric_chi_square_emanon3_uniform(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon3(P_UNI, Q_UNI), 0.0)

    def test_vicis_symmetric_chi_square_emanon3_p_zero_uses_epsilon(self):
        result = vicissitude.vicis_symmetric_chi_square_emanon3([0.0], [0.5])
        expected = (0.00001 - 0.5) ** 2 / 0.00001
        self.assertEqual(result, expected)

    # ── vicis_symmetric_chi_square_emanon4 ───────────────────────────────

    def test_vicis_symmetric_chi_square_emanon4(self):
        answer = sum([
            ((0.389 - 0.35) ** 2) / max([0.389, 0.35]), ((0.5 - 0.563) ** 2) / max([0.5, 0.563]),
            ((0.25 - 0.4) ** 2) / max([0.25, 0.4]), ((0.625 - 0.5) ** 2) / max([0.625, 0.5]),
            ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
            ((0.25 - 0.00001) ** 2) / max([0.25, 0.00001])
        ])
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4(p=P_STD, q=Q_STD), answer)

    def test_vicis_symmetric_chi_square_emanon4_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4(P_NZ, P_NZ), 0.0)

    def test_vicis_symmetric_chi_square_emanon4_single_identical(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4([0.5], [0.5]), 0.0)

    def test_vicis_symmetric_chi_square_emanon4_single_different(self):
        answer = (0.3 - 0.7) ** 2 / max(0.3, 0.7)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4([0.3], [0.7]), answer)

    def test_vicis_symmetric_chi_square_emanon4_both_zero_uses_epsilon(self):
        result = vicissitude.vicis_symmetric_chi_square_emanon4([0.0], [0.0])
        expected = (0.00001 - 0.00001) ** 2 / 0.00001
        self.assertEqual(result, expected)

    def test_vicis_symmetric_chi_square_emanon4_symmetric(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4(P_NZ, Q_NZ),
                         vicissitude.vicis_symmetric_chi_square_emanon4(Q_NZ, P_NZ))

    def test_vicis_symmetric_chi_square_emanon4_non_negative(self):
        self.assertGreaterEqual(vicissitude.vicis_symmetric_chi_square_emanon4(P_STD, Q_STD), 0.0)

    def test_vicis_symmetric_chi_square_emanon4_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        answer = (0.3 - 0.4) ** 2 / max(0.3, 0.4) + (0.7 - 0.6) ** 2 / max(0.7, 0.6)
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4(p, q), answer)

    def test_vicis_symmetric_chi_square_emanon4_uniform(self):
        self.assertEqual(vicissitude.vicis_symmetric_chi_square_emanon4(P_UNI, Q_UNI), 0.0)

    def test_vicis_symmetric_chi_square_emanon4_less_than_emanon3(self):
        # max denominator >= min denominator, so emanon4 <= emanon3
        e3 = vicissitude.vicis_symmetric_chi_square_emanon3(P_NZ, Q_NZ)
        e4 = vicissitude.vicis_symmetric_chi_square_emanon4(P_NZ, Q_NZ)
        self.assertLessEqual(e4, e3)

    # ── max_symmetric_chi_square_emanon5 ─────────────────────────────────

    def test_max_symmetric_chi_square_emanon5(self):
        answer_l = sum([((0.389 - 0.35) ** 2) / 0.389, ((0.5 - 0.563) ** 2) / 0.5,
                        ((0.25 - 0.4) ** 2) / 0.25, ((0.625 - 0.5) ** 2) / 0.625,
                        ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
                        ((0.25 - 0.00001) ** 2) / 0.25])
        answer_r = sum([((0.389 - 0.35) ** 2) / 0.35, ((0.5 - 0.563) ** 2) / 0.563,
                        ((0.25 - 0.4) ** 2) / 0.4, ((0.625 - 0.5) ** 2) / 0.5,
                        ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
                        ((0.25 - 0.00001) ** 2) / 0.00001])
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5(p=P_STD, q=Q_STD),
                         max([answer_l, answer_r]))

    def test_max_symmetric_chi_square_emanon5_identical(self):
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5(P_NZ, P_NZ), 0.0)

    def test_max_symmetric_chi_square_emanon5_single_identical(self):
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5([0.5], [0.5]), 0.0)

    def test_max_symmetric_chi_square_emanon5_single_different(self):
        left = (0.3 - 0.7) ** 2 / 0.3
        right = (0.3 - 0.7) ** 2 / 0.7
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5([0.3], [0.7]), max(left, right))

    def test_max_symmetric_chi_square_emanon5_both_zero_uses_epsilon(self):
        result = vicissitude.max_symmetric_chi_square_emanon5([0.0], [0.0])
        val = (0.00001 - 0.00001) ** 2 / 0.00001
        self.assertEqual(result, max(val, val))

    def test_max_symmetric_chi_square_emanon5_non_negative(self):
        self.assertGreaterEqual(vicissitude.max_symmetric_chi_square_emanon5(P_STD, Q_STD), 0.0)

    def test_max_symmetric_chi_square_emanon5_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        left = (0.3 - 0.4) ** 2 / 0.3 + (0.7 - 0.6) ** 2 / 0.7
        right = (0.3 - 0.4) ** 2 / 0.4 + (0.7 - 0.6) ** 2 / 0.6
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5(p, q), max(left, right))

    def test_max_symmetric_chi_square_emanon5_uniform(self):
        self.assertEqual(vicissitude.max_symmetric_chi_square_emanon5(P_UNI, Q_UNI), 0.0)

    def test_max_symmetric_chi_square_emanon5_greater_equal_min(self):
        max_val = vicissitude.max_symmetric_chi_square_emanon5(P_NZ, Q_NZ)
        min_val = vicissitude.min_symmetric_chi_square_emanon6(P_NZ, Q_NZ)
        self.assertGreaterEqual(max_val, min_val)

    def test_max_symmetric_chi_square_emanon5_greater_equal_neyman(self):
        from scikit_pierre.measures import chi
        result = vicissitude.max_symmetric_chi_square_emanon5(P_NZ, Q_NZ)
        neyman = chi.neyman_square(P_NZ, Q_NZ)
        person = chi.person_chi_square(P_NZ, Q_NZ)
        self.assertGreaterEqual(result, min(neyman, person))

    def test_max_symmetric_chi_square_emanon5_p_zero_uses_epsilon(self):
        result = vicissitude.max_symmetric_chi_square_emanon5([0.0], [0.5])
        left = (0.00001 - 0.5) ** 2 / 0.00001
        right = (0.00001 - 0.5) ** 2 / 0.5
        self.assertEqual(result, max(left, right))

    # ── min_symmetric_chi_square_emanon6 ─────────────────────────────────

    def test_min_symmetric_chi_square_emanon6(self):
        answer_l = sum([((0.389 - 0.35) ** 2) / 0.389, ((0.5 - 0.563) ** 2) / 0.5,
                        ((0.25 - 0.4) ** 2) / 0.25, ((0.625 - 0.5) ** 2) / 0.625,
                        ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
                        ((0.25 - 0.00001) ** 2) / 0.25])
        answer_r = sum([((0.389 - 0.35) ** 2) / 0.35, ((0.5 - 0.563) ** 2) / 0.563,
                        ((0.25 - 0.4) ** 2) / 0.4, ((0.625 - 0.5) ** 2) / 0.5,
                        ((0.00001 - 0.00001) ** 2) / 0.00001, ((0.00001 - 0.00001) ** 2) / 0.00001,
                        ((0.25 - 0.00001) ** 2) / 0.00001])
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6(p=P_STD, q=Q_STD),
                         min([answer_l, answer_r]))

    def test_min_symmetric_chi_square_emanon6_identical(self):
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6(P_NZ, P_NZ), 0.0)

    def test_min_symmetric_chi_square_emanon6_single_identical(self):
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6([0.5], [0.5]), 0.0)

    def test_min_symmetric_chi_square_emanon6_single_different(self):
        left = (0.3 - 0.7) ** 2 / 0.3
        right = (0.3 - 0.7) ** 2 / 0.7
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6([0.3], [0.7]), min(left, right))

    def test_min_symmetric_chi_square_emanon6_both_zero_uses_epsilon(self):
        result = vicissitude.min_symmetric_chi_square_emanon6([0.0], [0.0])
        val = (0.00001 - 0.00001) ** 2 / 0.00001
        self.assertEqual(result, min(val, val))

    def test_min_symmetric_chi_square_emanon6_non_negative(self):
        self.assertGreaterEqual(vicissitude.min_symmetric_chi_square_emanon6(P_STD, Q_STD), 0.0)

    def test_min_symmetric_chi_square_emanon6_two_elements(self):
        p, q = [0.3, 0.7], [0.4, 0.6]
        left = (0.3 - 0.4) ** 2 / 0.3 + (0.7 - 0.6) ** 2 / 0.7
        right = (0.3 - 0.4) ** 2 / 0.4 + (0.7 - 0.6) ** 2 / 0.6
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6(p, q), min(left, right))

    def test_min_symmetric_chi_square_emanon6_uniform(self):
        self.assertEqual(vicissitude.min_symmetric_chi_square_emanon6(P_UNI, Q_UNI), 0.0)

    def test_min_symmetric_chi_square_emanon6_less_equal_max(self):
        min_val = vicissitude.min_symmetric_chi_square_emanon6(P_NZ, Q_NZ)
        max_val = vicissitude.max_symmetric_chi_square_emanon5(P_NZ, Q_NZ)
        self.assertLessEqual(min_val, max_val)

    def test_min_symmetric_chi_square_emanon6_p_zero_uses_epsilon(self):
        result = vicissitude.min_symmetric_chi_square_emanon6([0.0], [0.5])
        left = (0.00001 - 0.5) ** 2 / 0.00001
        right = (0.00001 - 0.5) ** 2 / 0.5
        self.assertEqual(result, min(left, right))
