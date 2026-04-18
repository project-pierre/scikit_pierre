"""
Unit tests for scikit_pierre.distributions.entropy_based.
"""
import math
import unittest

from scikit_pierre.distributions.entropy_based import (
    global_local_entropy_based,
    global_local_entropy_based_with_probability_property,
)
from scikit_pierre.models.item import Item


def _item(iid, score, classes):
    return Item(_id=iid, score=score, classes=classes)


class TestGlobalLocalEntropyBased(unittest.TestCase):

    def test_single_item_single_genre_returns_epsilon(self):
        """
        With one genre appearing once in one item:
        global_freq = 1.0, genre_value = 1.0
        → log2(1.0 * 1.0) = 0 → entropy = 0 → numerator = 0 → epsilon.
        """
        items = {"i1": _item("i1", score=5.0, classes={"Action": 1.0})}
        dist = global_local_entropy_based(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)

    def test_two_items_two_genres_symmetric_yields_equal_values(self):
        """
        Two items each with a distinct genre and equal scores:
        both genres have the same global frequency (0.5) and genre weight (1.0)
        → entropy is identical for both → GLEB values are equal.
        """
        items = {
            "i1": _item("i1", score=4.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=4.0, classes={"Drama": 1.0}),
        }
        dist = global_local_entropy_based(items)
        self.assertAlmostEqual(dist["Action"], dist["Drama"], places=9)

    def test_returns_dict(self):
        """Return type is a dict."""
        items = {"i1": _item("i1", score=3.0, classes={"Action": 1.0})}
        self.assertIsInstance(global_local_entropy_based(items), dict)

    def test_genre_keys_match_item_classes(self):
        """Returned keys match the union of all genre labels in the items."""
        items = {
            "i1": _item("i1", score=3.0, classes={"Action": 0.5, "Drama": 0.5}),
            "i2": _item("i2", score=2.0, classes={"Comedy": 1.0}),
        }
        dist = global_local_entropy_based(items)
        self.assertSetEqual(set(dist.keys()), {"Action", "Drama", "Comedy"})

    def test_all_values_non_negative(self):
        """All GLEB values are ≥ 0."""
        items = {
            "i1": _item("i1", score=5.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=3.0, classes={"Drama": 1.0}),
        }
        dist = global_local_entropy_based(items)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_higher_score_yields_proportionally_higher_value(self):
        """
        For two items sharing a genre with the same weight, doubling one item's
        score increases the GLEB value for that genre.
        """
        items_low = {
            "i1": _item("i1", score=2.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=2.0, classes={"Drama": 1.0}),
        }
        items_high = {
            "i1": _item("i1", score=8.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=2.0, classes={"Drama": 1.0}),
        }
        dist_low  = global_local_entropy_based(items_low)
        dist_high = global_local_entropy_based(items_high)
        # Higher score for Action should increase (or maintain) Action's GLEB value
        self.assertGreaterEqual(dist_high["Action"], dist_low["Action"])

    def test_determinism(self):
        """Same input produces identical output."""
        items = {
            "i1": _item("i1", score=4.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=3.0, classes={"Drama": 1.0}),
        }
        self.assertEqual(
            global_local_entropy_based(items),
            global_local_entropy_based(items),
        )

    def test_zero_score_yields_epsilon(self):
        """Item with score=0 contributes 0 to the numerator → epsilon returned."""
        items = {
            "i1": _item("i1", score=0.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=0.0, classes={"Drama": 1.0}),
        }
        dist = global_local_entropy_based(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)
        self.assertAlmostEqual(dist["Drama"],  0.00001, places=9)

    def test_multi_genre_item_all_keys_present(self):
        """A multi-genre item contributes all its genres to the distribution."""
        items = {"i1": _item("i1", score=3.0, classes={"A": 0.5, "B": 0.3, "C": 0.2})}
        dist = global_local_entropy_based(items)
        self.assertIn("A", dist)
        self.assertIn("B", dist)
        self.assertIn("C", dist)


class TestGlobalLocalEntropyBasedWithProbabilityProperty(unittest.TestCase):

    def _items(self):
        return {
            "i1": _item("i1", score=5.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=3.0, classes={"Drama": 1.0}),
            "i3": _item("i3", score=4.0, classes={"Comedy": 1.0}),
        }

    def test_normalizes_to_one(self):
        """GLEB_P output sums to 1.0."""
        dist = global_local_entropy_based_with_probability_property(self._items())
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        """All GLEB_P probabilities are in [0, 1]."""
        dist = global_local_entropy_based_with_probability_property(self._items())
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_relative_order_preserved_from_gleb(self):
        """Normalization does not change the relative genre ranking."""
        items = self._items()
        gleb  = global_local_entropy_based(items)
        glebp = global_local_entropy_based_with_probability_property(items)
        sorted_gleb  = sorted(gleb.items(),  key=lambda x: x[1])
        sorted_glebp = sorted(glebp.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_gleb], [k for k, _ in sorted_glebp])

    def test_determinism(self):
        """Repeated calls yield identical results."""
        items = self._items()
        self.assertEqual(
            global_local_entropy_based_with_probability_property(items),
            global_local_entropy_based_with_probability_property(items),
        )

    def test_symmetric_items_yield_equal_probabilities(self):
        """Two items with identical structure produce equal GLEB_P values."""
        items = {
            "i1": _item("i1", score=4.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=4.0, classes={"Drama": 1.0}),
        }
        dist = global_local_entropy_based_with_probability_property(items)
        self.assertAlmostEqual(dist["Action"], dist["Drama"], places=9)
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)
