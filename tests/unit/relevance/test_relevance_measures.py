"""
Unit tests for scikit_pierre/relevance/relevance_measures.py
"""
import math
import unittest
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scikit_pierre.relevance.relevance_measures import (
    ndcg_relevance_score,
    relevance_tecrec,
    sum_relevance_score,
    utility_relevance_scores,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SCORES_MIXED = [4, 5, 4.5, 4, 5]
SCORES_UNIFORM = [5, 5, 5, 5, 5, 5]
SCORES_LARGE = [100, 80, 90, 50, 90]
SCORES_ZEROS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SCORES_ONES = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
SCORES_FRAC = [0.388, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]
SCORES_SINGLE = [3.7]
SCORES_DESCENDING = [5.0, 4.0, 3.0, 2.0, 1.0]
SCORES_ASCENDING = [1.0, 2.0, 3.0, 4.0, 5.0]


# ===========================================================================
# sum_relevance_score
# ===========================================================================

class TestSumRelevanceScore(unittest.TestCase):

    def test_sum_mixed_scores_equals_builtin_sum(self):
        """Golden: result equals Python's built-in sum."""
        self.assertEqual(sum_relevance_score(SCORES_MIXED), sum(SCORES_MIXED))

    def test_sum_uniform_scores_equals_builtin_sum(self):
        self.assertEqual(sum_relevance_score(SCORES_UNIFORM), sum(SCORES_UNIFORM))

    def test_sum_large_scores_equals_builtin_sum(self):
        self.assertEqual(sum_relevance_score(SCORES_LARGE), sum(SCORES_LARGE))

    def test_sum_zeros_returns_zero(self):
        """All-zero list must produce 0."""
        self.assertEqual(sum_relevance_score(SCORES_ZEROS), 0.0)

    def test_sum_ones_returns_list_length(self):
        """List of ones: sum == len."""
        self.assertEqual(sum_relevance_score(SCORES_ONES), float(len(SCORES_ONES)))

    def test_sum_fractional_scores_equals_builtin_sum(self):
        self.assertAlmostEqual(sum_relevance_score(SCORES_FRAC), sum(SCORES_FRAC), places=10)

    def test_sum_single_element_returns_element(self):
        """Single-element list: result equals that element."""
        self.assertAlmostEqual(sum_relevance_score(SCORES_SINGLE), SCORES_SINGLE[0], places=10)

    def test_sum_negative_scores(self):
        scores = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(sum_relevance_score(scores), -6.0, places=10)

    def test_sum_empty_list_returns_zero(self):
        self.assertEqual(sum_relevance_score([]), 0)

    def test_sum_order_invariant(self):
        """Sum does not depend on order."""
        a = [1.0, 2.0, 3.0]
        b = [3.0, 1.0, 2.0]
        self.assertEqual(sum_relevance_score(a), sum_relevance_score(b))

    def test_sum_returns_float_or_numeric(self):
        result = sum_relevance_score(SCORES_MIXED)
        self.assertIsInstance(result, (int, float))


@pytest.mark.parametrize("scores", [
    [4, 5, 4.5, 4, 5],
    [5, 5, 5, 5, 5, 5],
    [100, 80, 90, 50, 90],
    [0.0, 0.0],
    [1.0],
    [0.388, 0.5, 0.25],
])
def test_sum_relevance_parametrized(scores):
    """sum_relevance_score always equals built-in sum."""
    assert sum_relevance_score(scores) == pytest.approx(sum(scores))


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=50))
@settings(max_examples=200)
def test_sum_relevance_property_equals_builtin(scores):
    assert sum_relevance_score(scores) == pytest.approx(sum(scores), rel=1e-9)


# ===========================================================================
# ndcg_relevance_score
# ===========================================================================

def _reference_ndcg(scores):
    """Reference DCG implementation used in existing tests (no rounding)."""
    if not scores:
        return 0.0
    dcg = sum(((2 ** w) - 1) / (np.log2(i + 2)) for i, w in enumerate(scores))
    idcg = sum(((2 ** w) - 1) / (np.log2(i + 2)) for i, w in enumerate(sorted(scores, reverse=True)))
    if not idcg:
        return 0.0
    return dcg / idcg


class TestNdcgRelevanceScore(unittest.TestCase):

    def test_ndcg_none_returns_zero(self):
        """None input must return 0.0."""
        self.assertEqual(ndcg_relevance_score(None), 0.0)

    def test_ndcg_empty_list_returns_zero(self):
        self.assertEqual(ndcg_relevance_score([]), 0.0)

    def test_ndcg_all_zeros_returns_zero(self):
        """All-zero relevances: DCG = IDCG = 0, must return 0."""
        self.assertEqual(ndcg_relevance_score(SCORES_ZEROS), 0.0)

    def test_ndcg_uniform_scores_returns_one(self):
        """Identical scores: every ordering is ideal, NDCG = 1.0."""
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_UNIFORM), 1.0, places=4)

    def test_ndcg_ones_returns_one(self):
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_ONES), 1.0, places=4)

    def test_ndcg_mixed_scores_bounded(self):
        result = ndcg_relevance_score(SCORES_MIXED)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_ndcg_descending_sorted_list_returns_one(self):
        """Already-sorted descending list achieves maximum NDCG = 1.0."""
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_DESCENDING), 1.0, places=4)

    def test_ndcg_ascending_sorted_list_below_one(self):
        """Worst-case ordering must score below 1.0."""
        result = ndcg_relevance_score(SCORES_ASCENDING)
        self.assertLess(result, 1.0)

    def test_ndcg_mixed_matches_reference(self):
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_MIXED),
                               _reference_ndcg(SCORES_MIXED), places=3)

    def test_ndcg_large_scores_matches_reference(self):
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_LARGE),
                               _reference_ndcg(SCORES_LARGE), places=3)

    def test_ndcg_fractional_scores_matches_reference(self):
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_FRAC),
                               _reference_ndcg(SCORES_FRAC), places=3)

    def test_ndcg_single_element_returns_one(self):
        """Single-element list: DCG == IDCG, result must be 1.0."""
        self.assertAlmostEqual(ndcg_relevance_score(SCORES_SINGLE), 1.0, places=4)

    def test_ndcg_returns_float(self):
        self.assertIsInstance(ndcg_relevance_score(SCORES_MIXED), float)


@pytest.mark.parametrize("scores,expected", [
    (None, 0.0),
    ([], 0.0),
    ([0.0, 0.0, 0.0], 0.0),
    ([1.0, 1.0, 1.0], 1.0),
    ([5.0, 4.0, 3.0, 2.0, 1.0], 1.0),
])
def test_ndcg_parametrized_known_values(scores, expected):
    assert ndcg_relevance_score(scores) == pytest.approx(expected, abs=1e-4)


@given(st.lists(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=30))
@settings(max_examples=200)
def test_ndcg_property_bounded_zero_to_one(scores):
    """NDCG is always in [0, 1]."""
    result = ndcg_relevance_score(scores)
    assert 0.0 <= result <= 1.0 + 1e-9


@given(st.lists(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=30))
@settings(max_examples=200)
def test_ndcg_property_sorted_desc_is_max(scores):
    """Descending-sorted list should achieve NDCG = 1.0."""
    sorted_scores = sorted(scores, reverse=True)
    result = ndcg_relevance_score(sorted_scores)
    assert result == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# utility_relevance_scores
# ===========================================================================

def _reference_utility(scores):
    """Expected utility: weighted sum where weight = 1/log(position+1), 1-indexed."""
    def _u(values):
        return sum(v * (1.0 / math.log(ix + 1)) for ix, v in enumerate(values, start=1))
    s = _u(scores)
    ideal = _u(sorted(scores, reverse=True))
    if not ideal:
        return 0.0
    return s / ideal


class TestUtilityRelevanceScores(unittest.TestCase):

    def test_utility_uniform_scores_returns_one(self):
        """Identical scores: every ordering is ideal, utility = 1.0."""
        self.assertAlmostEqual(utility_relevance_scores(SCORES_UNIFORM), 1.0, places=6)

    def test_utility_ones_returns_one(self):
        self.assertAlmostEqual(utility_relevance_scores(SCORES_ONES), 1.0, places=6)

    def test_utility_descending_returns_one(self):
        """Already-sorted descending achieves maximum utility = 1.0."""
        self.assertAlmostEqual(utility_relevance_scores(SCORES_DESCENDING), 1.0, places=6)

    def test_utility_ascending_below_one(self):
        result = utility_relevance_scores(SCORES_ASCENDING)
        self.assertLess(result, 1.0)

    def test_utility_mixed_matches_reference(self):
        self.assertAlmostEqual(utility_relevance_scores(SCORES_MIXED),
                               _reference_utility(SCORES_MIXED), places=6)

    def test_utility_large_scores_matches_reference(self):
        self.assertAlmostEqual(utility_relevance_scores(SCORES_LARGE),
                               _reference_utility(SCORES_LARGE), places=6)

    def test_utility_all_zeros_returns_zero(self):
        """All-zero scores: ideal is also 0, must return 0.0 (not ZeroDivisionError)."""
        self.assertEqual(utility_relevance_scores(SCORES_ZEROS), 0.0)

    def test_utility_bounded_above_by_one(self):
        """Utility must not exceed 1.0."""
        self.assertLessEqual(utility_relevance_scores(SCORES_MIXED), 1.0 + 1e-9)

    def test_utility_non_negative(self):
        self.assertGreaterEqual(utility_relevance_scores(SCORES_MIXED), 0.0)

    def test_utility_single_element_returns_one(self):
        self.assertAlmostEqual(utility_relevance_scores(SCORES_SINGLE), 1.0, places=6)

    def test_utility_values_actually_used_regression(self):
        """Regression: bug where 'value' was never used made all results equal 1.0.
        Two lists with different values but same length must produce different results."""
        high_first = [10.0, 1.0, 1.0, 1.0]
        low_first = [1.0, 1.0, 1.0, 10.0]
        self.assertGreater(utility_relevance_scores(high_first),
                           utility_relevance_scores(low_first))

    def test_utility_fractional_scores_matches_reference(self):
        scores = [0.9, 0.5, 0.1, 0.3]
        self.assertAlmostEqual(utility_relevance_scores(scores),
                               _reference_utility(scores), places=6)

    def test_utility_returns_float(self):
        self.assertIsInstance(utility_relevance_scores(SCORES_MIXED), float)


@pytest.mark.parametrize("scores", [
    [4, 5, 4.5, 4, 5],
    [100, 80, 90, 50, 90],
    [0.388, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
    [1.0],
    [3.0, 3.0, 3.0],
])
def test_utility_parametrized_matches_reference(scores):
    assert utility_relevance_scores(scores) == pytest.approx(_reference_utility(scores), rel=1e-5)


@given(st.lists(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=30))
@settings(max_examples=200)
def test_utility_property_bounded_zero_to_one(scores):
    """Utility is always in [0, 1]."""
    result = utility_relevance_scores(scores)
    assert 0.0 <= result <= 1.0 + 1e-9


@given(st.lists(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=30))
@settings(max_examples=200)
def test_utility_property_sorted_desc_is_max(scores):
    """Descending-sorted list achieves maximum utility = 1.0."""
    sorted_scores = sorted(scores, reverse=True)
    assert utility_relevance_scores(sorted_scores) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# relevance_tecrec
# ===========================================================================

def _reference_tecrec(scores):
    return sum(d / (len(scores) + 1) for d in scores)


class TestRelevanceTecRec(unittest.TestCase):

    def test_tecrec_mixed_scores_matches_reference(self):
        self.assertAlmostEqual(relevance_tecrec(SCORES_MIXED),
                               _reference_tecrec(SCORES_MIXED), places=10)

    def test_tecrec_uniform_scores_matches_reference(self):
        self.assertAlmostEqual(relevance_tecrec(SCORES_UNIFORM),
                               _reference_tecrec(SCORES_UNIFORM), places=10)

    def test_tecrec_large_scores_matches_reference(self):
        self.assertAlmostEqual(relevance_tecrec(SCORES_LARGE),
                               _reference_tecrec(SCORES_LARGE), places=10)

    def test_tecrec_zeros_returns_zero(self):
        self.assertAlmostEqual(relevance_tecrec(SCORES_ZEROS), 0.0, places=10)

    def test_tecrec_single_element(self):
        """Single-element: result = score / 2."""
        score = 4.0
        self.assertAlmostEqual(relevance_tecrec([score]), score / 2.0, places=10)

    def test_tecrec_equals_sum_divided_by_n_plus_one(self):
        """Closed-form: tecrec == sum(scores) / (len(scores) + 1)."""
        scores = [3.0, 1.0, 4.0, 1.0, 5.0]
        expected = sum(scores) / (len(scores) + 1)
        self.assertAlmostEqual(relevance_tecrec(scores), expected, places=10)

    def test_tecrec_fractional_scores_matches_reference(self):
        self.assertAlmostEqual(relevance_tecrec(SCORES_FRAC),
                               _reference_tecrec(SCORES_FRAC), places=10)

    def test_tecrec_order_invariant(self):
        """TecRec is order-invariant (only depends on sum and length)."""
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(relevance_tecrec(a), relevance_tecrec(b), places=10)

    def test_tecrec_non_negative_for_non_negative_input(self):
        self.assertGreaterEqual(relevance_tecrec(SCORES_FRAC), 0.0)

    def test_tecrec_negative_scores(self):
        scores = [-2.0, -1.0]
        self.assertAlmostEqual(relevance_tecrec(scores), _reference_tecrec(scores), places=10)

    def test_tecrec_longer_list_smaller_per_item_contribution(self):
        """Longer list → larger denominator → each item contributes less."""
        scores = [1.0]
        weight_2 = relevance_tecrec([1.0, 0.0])   # 1/(2+1) = 1/3
        weight_4 = relevance_tecrec([1.0, 0.0, 0.0, 0.0])  # 1/(4+1) = 1/5
        self.assertGreater(weight_2, weight_4)

    def test_tecrec_returns_float(self):
        self.assertIsInstance(relevance_tecrec(SCORES_MIXED), float)


@pytest.mark.parametrize("scores", [
    [4, 5, 4.5, 4, 5],
    [5, 5, 5, 5, 5, 5],
    [100, 80, 90, 50, 90],
    [0.0, 0.0, 0.0],
    [1.0],
    [0.388, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
])
def test_tecrec_parametrized(scores):
    assert relevance_tecrec(scores) == pytest.approx(_reference_tecrec(scores), rel=1e-9)


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=50))
@settings(max_examples=200)
def test_tecrec_property_equals_sum_over_n_plus_one(scores):
    """TecRec always equals sum(scores) / (len(scores) + 1)."""
    expected = sum(scores) / (len(scores) + 1)
    assert relevance_tecrec(scores) == pytest.approx(expected, rel=1e-9)


if __name__ == "__main__":
    unittest.main()
