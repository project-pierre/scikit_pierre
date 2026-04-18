"""
Unit tests for scikit_pierre.distributions.compute_tilde_q.
"""
import math
import unittest

import pytest

from scikit_pierre.distributions.compute_tilde_q import compute_tilde_q


class TestComputeTildeQ(unittest.TestCase):

    # ── empty input ────────────────────────────────────────────────────────

    def test_empty_lists_return_empty(self):
        """Empty p and q yield an empty result."""
        self.assertEqual(compute_tilde_q([], []), [])

    # ── single-element golden values ───────────────────────────────────────

    def test_single_element_default_alpha(self):
        """tilde_q = (1-0.01)*q + 0.01*p  → 0.99*0 + 0.01*1 = 0.01."""
        result = compute_tilde_q([1.0], [0.0])
        self.assertAlmostEqual(result[0], 0.01, places=10)

    def test_single_element_reversed(self):
        """tilde_q with p=0, q=1 → 0.99*1 + 0.01*0 = 0.99."""
        result = compute_tilde_q([0.0], [1.0])
        self.assertAlmostEqual(result[0], 0.99, places=10)

    # ── boundary alpha values ──────────────────────────────────────────────

    def test_alpha_zero_returns_q(self):
        """alpha=0 → result equals q exactly."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        result = compute_tilde_q(p, q, alpha=0.0)
        for r, qv in zip(result, q):
            self.assertAlmostEqual(r, qv, places=10)

    def test_alpha_one_returns_p(self):
        """alpha=1 → result equals p exactly."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        result = compute_tilde_q(p, q, alpha=1.0)
        for r, pv in zip(result, p):
            self.assertAlmostEqual(r, pv, places=10)

    # ── identical distributions ────────────────────────────────────────────

    def test_p_equals_q_returns_p(self):
        """When p == q, tilde_q == p regardless of alpha."""
        p = [0.2, 0.5, 0.3]
        result = compute_tilde_q(p, p[:])
        for r, pv in zip(result, p):
            self.assertAlmostEqual(r, pv, places=10)

    # ── normalization ──────────────────────────────────────────────────────

    def test_normalized_inputs_yield_normalized_output(self):
        """If both p and q sum to 1.0, tilde_q also sums to 1.0."""
        p = [0.2, 0.5, 0.3]
        q = [0.4, 0.1, 0.5]
        result = compute_tilde_q(p, q)
        self.assertAlmostEqual(sum(result), 1.0, places=9)

    def test_normalization_holds_for_various_alphas(self):
        """Normalization is preserved for any alpha in [0, 1]."""
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.1, 0.4, 0.3, 0.2]
        for alpha in [0.0, 0.01, 0.1, 0.5, 1.0]:
            result = compute_tilde_q(p, q, alpha=alpha)
            self.assertAlmostEqual(sum(result), 1.0, places=9,
                                   msg=f"Failed for alpha={alpha}")

    # ── non-negativity ─────────────────────────────────────────────────────

    def test_non_negative_when_inputs_non_negative(self):
        """All tilde_q values are ≥ 0 when p, q ≥ 0."""
        p = [0.0, 0.5, 0.5]
        q = [0.3, 0.3, 0.4]
        result = compute_tilde_q(p, q)
        for v in result:
            self.assertGreaterEqual(v, 0.0)

    def test_zero_mass_in_q_gets_smoothed(self):
        """A zero entry in q receives positive mass from p via alpha."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        result = compute_tilde_q(p, q, alpha=0.01)
        # First element: 0.99*0 + 0.01*1 = 0.01 > 0
        self.assertGreater(result[0], 0.0)

    # ── output length ──────────────────────────────────────────────────────

    def test_output_length_matches_input(self):
        """Result list has the same length as the input lists."""
        p = [0.1, 0.2, 0.3, 0.4]
        q = [0.4, 0.3, 0.2, 0.1]
        self.assertEqual(len(compute_tilde_q(p, q)), 4)

    # ── interpolation correctness ──────────────────────────────────────────

    def test_element_is_convex_combination_of_p_and_q(self):
        """Each element lies strictly between the p and q values for alpha in (0,1)."""
        p = [0.8, 0.2]
        q = [0.3, 0.7]
        alpha = 0.1
        result = compute_tilde_q(p, q, alpha=alpha)
        expected = [(1 - alpha) * qv + alpha * pv for pv, qv in zip(p, q)]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=10)

    def test_golden_value_two_elements(self):
        """Hand-verified: p=[0.6,0.4], q=[0.9,0.1], alpha=0.01."""
        result = compute_tilde_q([0.6, 0.4], [0.9, 0.1], alpha=0.01)
        self.assertAlmostEqual(result[0], 0.99 * 0.9 + 0.01 * 0.6, places=10)
        self.assertAlmostEqual(result[1], 0.99 * 0.1 + 0.01 * 0.4, places=10)

    # ── determinism ───────────────────────────────────────────────────────

    def test_determinism_same_output_on_repeated_calls(self):
        """Identical inputs always produce identical outputs."""
        p = [0.3, 0.4, 0.3]
        q = [0.1, 0.7, 0.2]
        r1 = compute_tilde_q(p, q)
        r2 = compute_tilde_q(p, q)
        self.assertEqual(r1, r2)
