"""
Unit tests for scikit_pierre/relevance/accessible.py (relevance_measures_funcs dispatcher).
"""
import unittest

import pytest

from scikit_pierre.relevance import accessible
from scikit_pierre.relevance import relevance_measures


class TestRelevanceMeasuresFuncs(unittest.TestCase):

    def test_returns_sum_by_default(self):
        """Default argument 'SUM' must return sum_relevance_score."""
        self.assertIs(accessible.relevance_measures_funcs("SUM"),
                      relevance_measures.sum_relevance_score)

    def test_returns_ndcg(self):
        self.assertIs(accessible.relevance_measures_funcs("NDCG"),
                      relevance_measures.ndcg_relevance_score)

    def test_returns_urel(self):
        self.assertIs(accessible.relevance_measures_funcs("UREL"),
                      relevance_measures.utility_relevance_scores)

    def test_returns_tecrec(self):
        self.assertIs(accessible.relevance_measures_funcs("TECREC"),
                      relevance_measures.relevance_tecrec)

    def test_invalid_key_raises_name_error(self):
        with self.assertRaises(NameError):
            accessible.relevance_measures_funcs("INVALID_KEY")

    def test_invalid_key_error_message_contains_name(self):
        try:
            accessible.relevance_measures_funcs("UNKNOWN_XYZ")
        except NameError as e:
            self.assertIn("UNKNOWN_XYZ", str(e))

    def test_empty_string_raises_name_error(self):
        with self.assertRaises(NameError):
            accessible.relevance_measures_funcs("")

    def test_lowercase_sum_raises_name_error(self):
        """Keys are case-sensitive; 'sum' must not match 'SUM'."""
        with self.assertRaises(NameError):
            accessible.relevance_measures_funcs("sum")

    def test_returned_function_is_callable(self):
        for key in ("SUM", "NDCG", "UREL", "TECREC"):
            with self.subTest(key=key):
                self.assertTrue(callable(accessible.relevance_measures_funcs(key)))

    def test_sum_function_produces_correct_result(self):
        func = accessible.relevance_measures_funcs("SUM")
        scores = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(func(scores), 6.0, places=10)

    def test_ndcg_function_produces_float_in_range(self):
        func = accessible.relevance_measures_funcs("NDCG")
        result = func([4.0, 5.0, 3.0, 2.0])
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_urel_function_produces_float_in_range(self):
        func = accessible.relevance_measures_funcs("UREL")
        result = func([4.0, 5.0, 3.0, 2.0])
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0 + 1e-9)

    def test_tecrec_function_produces_numeric(self):
        func = accessible.relevance_measures_funcs("TECREC")
        result = func([1.0, 2.0, 3.0])
        self.assertIsInstance(result, (int, float))


@pytest.mark.parametrize("key,expected_fn", [
    ("SUM", relevance_measures.sum_relevance_score),
    ("NDCG", relevance_measures.ndcg_relevance_score),
    ("UREL", relevance_measures.utility_relevance_scores),
    ("TECREC", relevance_measures.relevance_tecrec),
])
def test_dispatcher_parametrized(key, expected_fn):
    assert accessible.relevance_measures_funcs(key) is expected_fn


@pytest.mark.parametrize("invalid_key", ["", "sum", "ndcg", "UNKNOWN", "SUM ", " SUM"])
def test_invalid_keys_raise_name_error(invalid_key):
    with pytest.raises(NameError):
        accessible.relevance_measures_funcs(invalid_key)


if __name__ == "__main__":
    unittest.main()
