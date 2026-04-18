"""
Unit tests for scikit_pierre.distributions.time_slide_window_based.

Covers: temporal_slide_window_base_function and all wrapper functions
(temporal_slide_window, temporal_slide_window_with_probability_property,
mixed_tsw_gleb, mixed_tsw_twb, mixed_tsw_twb_gleb, and their _P variants).
"""
import unittest

from scikit_pierre.distributions.time_slide_window_based import (
    mixed_tsw_gleb,
    mixed_tsw_gleb_with_probability_property,
    mixed_tsw_twb,
    mixed_tsw_twb_gleb,
    mixed_tsw_twb_gleb_with_probability_property,
    mixed_tsw_twb_with_probability_property,
    temporal_slide_window,
    temporal_slide_window_base_function,
    temporal_slide_window_base_function_with_probability_property,
    temporal_slide_window_with_probability_property,
)
from scikit_pierre.models.item import Item


def _item(iid, score, time, classes):
    return Item(_id=iid, score=score, time=time, classes=classes)


def _item_no_time(iid, score, classes):
    """Item without time attribute (uses default None → 0.0 in TWB)."""
    return Item(_id=iid, score=score, classes=classes)


# ---------------------------------------------------------------------------
# Small fixture (≤20 items, so window fallback path)
# ---------------------------------------------------------------------------

ITEMS_SMALL = {
    f"i{i}": _item(f"i{i}", score=float(i + 1), time=float(i + 1) / 10,
                   classes={"Action": 0.5, "Drama": 0.5})
    for i in range(5)
}

# Single-genre small fixture.
ITEMS_SINGLE_GENRE = {
    f"i{i}": _item(f"i{i}", score=1.0, time=0.5, classes={"Action": 1.0})
    for i in range(5)
}


def _make_large_items(n: int) -> dict:
    """Create n items alternating between Action and Drama genres."""
    items = {}
    for i in range(n):
        genre = "Action" if i % 2 == 0 else "Drama"
        items[f"i{i}"] = _item(f"i{i}", score=float(i + 1), time=float(i % 5 + 1) / 5,
                                classes={genre: 1.0})
    return items


# Large fixture (>20 items, triggers windowing path).
ITEMS_LARGE = _make_large_items(25)


# ===========================================================================
# temporal_slide_window_base_function — fallback path (≤20 items)
# ===========================================================================

class TestTemporalSlideWindowBaseFunctionSmall(unittest.TestCase):

    def test_returns_dict(self):
        """Return type is a dict."""
        dist = temporal_slide_window_base_function(ITEMS_SMALL)
        self.assertIsInstance(dist, dict)

    def test_genre_keys_match_items(self):
        """Keys are the union of genres found in the item set."""
        dist = temporal_slide_window_base_function(ITEMS_SMALL)
        self.assertSetEqual(set(dist.keys()), {"Action", "Drama"})

    def test_all_values_non_negative(self):
        """All distribution values are ≥ 0."""
        dist = temporal_slide_window_base_function(ITEMS_SMALL)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_determinism(self):
        """Same input always produces identical output."""
        d1 = temporal_slide_window_base_function(ITEMS_SMALL)
        d2 = temporal_slide_window_base_function(ITEMS_SMALL)
        self.assertEqual(d1, d2)

    def test_single_genre_produces_single_key(self):
        """Single-genre item set yields exactly one genre key."""
        dist = temporal_slide_window_base_function(ITEMS_SINGLE_GENRE)
        self.assertSetEqual(set(dist.keys()), {"Action"})

    def test_small_set_falls_back_to_base_distribution(self):
        """With ≤20 items and major=10, the fallback path applies CWS directly."""
        from scikit_pierre.distributions.class_based import class_weighted_strategy
        expected = class_weighted_strategy(ITEMS_SMALL)
        dist = temporal_slide_window_base_function(ITEMS_SMALL, major=10, using="CWS")
        self.assertEqual(dist, expected)

    def test_using_gleb_fallback(self):
        """using='GLEB' falls back to global_local_entropy_based on small sets."""
        from scikit_pierre.distributions.entropy_based import global_local_entropy_based
        expected = global_local_entropy_based(ITEMS_SMALL)
        dist = temporal_slide_window_base_function(ITEMS_SMALL, major=10, using="GLEB")
        self.assertEqual(dist, expected)

    def test_using_twb_fallback(self):
        """using='TWB' falls back to time_weighted_based on small sets."""
        from scikit_pierre.distributions.time_based import time_weighted_based
        expected = time_weighted_based(ITEMS_SMALL)
        dist = temporal_slide_window_base_function(ITEMS_SMALL, major=10, using="TWB")
        self.assertEqual(dist, expected)

    def test_using_gleb_twb_fallback(self):
        """using='GLEB_TWB' falls back to mixed_gleb_twb on small sets."""
        from scikit_pierre.distributions.mixed_based import mixed_gleb_twb
        expected = mixed_gleb_twb(ITEMS_SMALL)
        dist = temporal_slide_window_base_function(ITEMS_SMALL, major=10, using="GLEB_TWB")
        self.assertEqual(dist, expected)

    def test_unknown_using_falls_back_to_cws(self):
        """An unrecognized using value defaults to the CWS base distribution."""
        from scikit_pierre.distributions.class_based import class_weighted_strategy
        expected = class_weighted_strategy(ITEMS_SMALL)
        dist = temporal_slide_window_base_function(ITEMS_SMALL, major=10, using="UNKNOWN")
        self.assertEqual(dist, expected)


# ===========================================================================
# temporal_slide_window_base_function — windowing path (>20 items)
# ===========================================================================

class TestTemporalSlideWindowBaseFunctionLarge(unittest.TestCase):

    def test_returns_dict(self):
        """Return type is a dict on the windowing path."""
        dist = temporal_slide_window_base_function(ITEMS_LARGE)
        self.assertIsInstance(dist, dict)

    def test_genre_keys_present(self):
        """Both Action and Drama appear in the distribution."""
        dist = temporal_slide_window_base_function(ITEMS_LARGE)
        self.assertIn("Action", dist)
        self.assertIn("Drama", dist)

    def test_all_values_non_negative(self):
        """All distribution values are ≥ 0 on the windowing path."""
        dist = temporal_slide_window_base_function(ITEMS_LARGE)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_determinism(self):
        """Same input produces identical output on windowing path."""
        d1 = temporal_slide_window_base_function(ITEMS_LARGE)
        d2 = temporal_slide_window_base_function(ITEMS_LARGE)
        self.assertEqual(d1, d2)

    def test_major_one_triggers_windowing_at_two_items(self):
        """major=1 triggers windowing for any dict with > 2 items."""
        items = _make_large_items(5)
        dist = temporal_slide_window_base_function(items, major=1, using="CWS")
        self.assertIsInstance(dist, dict)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_windowing_different_from_no_windowing(self):
        """On large sets, windowed result may differ from direct CWS (no strict equality required)."""
        from scikit_pierre.distributions.class_based import class_weighted_strategy
        direct = class_weighted_strategy(ITEMS_LARGE)
        windowed = temporal_slide_window_base_function(ITEMS_LARGE)
        # Both are valid dicts; they need not be equal but must share keys.
        self.assertSetEqual(set(direct.keys()), set(windowed.keys()))


# ===========================================================================
# temporal_slide_window_base_function_with_probability_property
# ===========================================================================

class TestTemporalSlideWindowBaseFunctionWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one_small(self):
        """TSW_P sums to 1.0 on the fallback path."""
        dist = temporal_slide_window_base_function_with_probability_property(ITEMS_SMALL)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_normalizes_to_one_large(self):
        """TSW_P sums to 1.0 on the windowing path."""
        dist = temporal_slide_window_base_function_with_probability_property(ITEMS_LARGE)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        """All TSW_P probabilities are in [0, 1]."""
        dist = temporal_slide_window_base_function_with_probability_property(ITEMS_SMALL)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_relative_order_preserved(self):
        """Normalization preserves relative genre ranking."""
        base = temporal_slide_window_base_function(ITEMS_SMALL)
        prob = temporal_slide_window_base_function_with_probability_property(ITEMS_SMALL)
        sorted_base = sorted(base.items(), key=lambda x: x[1])
        sorted_prob = sorted(prob.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_base], [k for k, _ in sorted_prob])

    def test_determinism(self):
        """Repeated calls yield identical results."""
        d1 = temporal_slide_window_base_function_with_probability_property(ITEMS_SMALL)
        d2 = temporal_slide_window_base_function_with_probability_property(ITEMS_SMALL)
        self.assertEqual(d1, d2)


# ===========================================================================
# Wrapper functions: temporal_slide_window / _with_probability_property
# ===========================================================================

class TestTemporalSlideWindow(unittest.TestCase):

    def test_equals_base_function_with_cws(self):
        """temporal_slide_window delegates to base function with using='CWS'."""
        expected = temporal_slide_window_base_function(ITEMS_SMALL, using="CWS")
        dist = temporal_slide_window(ITEMS_SMALL)
        self.assertEqual(dist, expected)

    def test_returns_dict(self):
        dist = temporal_slide_window(ITEMS_SMALL)
        self.assertIsInstance(dist, dict)

    def test_all_values_non_negative(self):
        for v in temporal_slide_window(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)


class TestTemporalSlideWindowWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        """TSW_P wrapper sums to 1.0."""
        dist = temporal_slide_window_with_probability_property(ITEMS_SMALL)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_equals_base_probability_function(self):
        """Wrapper equals the base _with_probability_property function."""
        expected = temporal_slide_window_base_function_with_probability_property(
            ITEMS_SMALL, using="CWS"
        )
        dist = temporal_slide_window_with_probability_property(ITEMS_SMALL)
        self.assertEqual(dist, expected)


# ===========================================================================
# mixed_tsw_gleb / mixed_tsw_gleb_with_probability_property
# ===========================================================================

class TestMixedTswGleb(unittest.TestCase):

    def test_equals_base_function_with_gleb(self):
        """mixed_tsw_gleb delegates to base function with using='GLEB'."""
        expected = temporal_slide_window_base_function(ITEMS_SMALL, using="GLEB")
        dist = mixed_tsw_gleb(ITEMS_SMALL)
        self.assertEqual(dist, expected)

    def test_returns_dict(self):
        self.assertIsInstance(mixed_tsw_gleb(ITEMS_SMALL), dict)

    def test_all_values_non_negative(self):
        for v in mixed_tsw_gleb(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)


class TestMixedTswGlebWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        dist = mixed_tsw_gleb_with_probability_property(ITEMS_SMALL)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        for v in mixed_tsw_gleb_with_probability_property(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ===========================================================================
# mixed_tsw_twb / mixed_tsw_twb_with_probability_property
# ===========================================================================

class TestMixedTswTwb(unittest.TestCase):

    def test_equals_base_function_with_twb(self):
        """mixed_tsw_twb delegates to base function with using='TWB'."""
        expected = temporal_slide_window_base_function(ITEMS_SMALL, using="TWB")
        dist = mixed_tsw_twb(ITEMS_SMALL)
        self.assertEqual(dist, expected)

    def test_returns_dict(self):
        self.assertIsInstance(mixed_tsw_twb(ITEMS_SMALL), dict)

    def test_all_values_non_negative(self):
        for v in mixed_tsw_twb(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)


class TestMixedTswTwbWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        dist = mixed_tsw_twb_with_probability_property(ITEMS_SMALL)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        for v in mixed_tsw_twb_with_probability_property(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ===========================================================================
# mixed_tsw_twb_gleb / mixed_tsw_twb_gleb_with_probability_property
# ===========================================================================

class TestMixedTswTwbGleb(unittest.TestCase):

    def test_equals_base_function_with_gleb_twb(self):
        """mixed_tsw_twb_gleb delegates to base function with using='GLEB_TWB'."""
        expected = temporal_slide_window_base_function(ITEMS_SMALL, using="GLEB_TWB")
        dist = mixed_tsw_twb_gleb(ITEMS_SMALL)
        self.assertEqual(dist, expected)

    def test_returns_dict(self):
        self.assertIsInstance(mixed_tsw_twb_gleb(ITEMS_SMALL), dict)

    def test_all_values_non_negative(self):
        for v in mixed_tsw_twb_gleb(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)


class TestMixedTswTwbGlebWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        dist = mixed_tsw_twb_gleb_with_probability_property(ITEMS_SMALL)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        for v in mixed_tsw_twb_gleb_with_probability_property(ITEMS_SMALL).values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
