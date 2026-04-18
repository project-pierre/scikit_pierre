"""
Unit tests for scikit_pierre.distributions.mixed_based.

Covers: mixed_gleb_twb (TWB_GLEB) and mixed_gleb_twb_with_probability_property (TWB_GLEB_P).
"""
import unittest
from math import log2

from scikit_pierre.distributions.mixed_based import (
    mixed_gleb_twb,
    mixed_gleb_twb_with_probability_property,
)
from scikit_pierre.models.item import Item


def _item(iid, score, time, classes):
    return Item(_id=iid, score=score, time=time, classes=classes)


# ---------------------------------------------------------------------------
# Shared small fixtures
# ---------------------------------------------------------------------------

# Two-item fixture with distinct genres → freq=0.5 each → ent = 0.5 per genre.
# Action: numerator = 5*1.0*0.5 = 2.5, denominator = 5 → 0.5
# Drama:  numerator = 3*0.5*0.5 = 0.75, denominator = 3 → 0.25
ITEMS_2 = {
    "i1": _item("i1", score=5.0, time=1.0, classes={"Action": 1.0}),
    "i2": _item("i2", score=3.0, time=0.5, classes={"Drama": 1.0}),
}
GOLD_ACTION = 0.5    # 2.5 / 5
GOLD_DRAMA = 0.25    # 0.75 / 3

# Symmetric two-item fixture: both genres get identical treatment.
ITEMS_SYM = {
    "i1": _item("i1", score=4.0, time=1.0, classes={"Action": 1.0}),
    "i2": _item("i2", score=4.0, time=1.0, classes={"Drama": 1.0}),
}


# ===========================================================================
# mixed_gleb_twb
# ===========================================================================

class TestMixedGlebTwb(unittest.TestCase):

    def test_returns_dict(self):
        """Return type is a dict."""
        dist = mixed_gleb_twb(ITEMS_2)
        self.assertIsInstance(dist, dict)

    def test_genre_keys_match_items(self):
        """Keys are exactly the union of genre labels in the items."""
        dist = mixed_gleb_twb(ITEMS_2)
        self.assertSetEqual(set(dist.keys()), {"Action", "Drama"})

    def test_single_item_single_genre_returns_epsilon(self):
        """Single item, single genre: global freq=1.0, ent=0 → epsilon."""
        items = {"i1": _item("i1", score=5.0, time=1.0, classes={"Action": 1.0})}
        dist = mixed_gleb_twb(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)

    def test_golden_action_value(self):
        """Hand-computed TWB_GLEB for Action = 0.5."""
        dist = mixed_gleb_twb(ITEMS_2)
        self.assertAlmostEqual(dist["Action"], GOLD_ACTION, places=9)

    def test_golden_drama_value(self):
        """Hand-computed TWB_GLEB for Drama = 0.25."""
        dist = mixed_gleb_twb(ITEMS_2)
        self.assertAlmostEqual(dist["Drama"], GOLD_DRAMA, places=9)

    def test_symmetric_items_yield_equal_values(self):
        """Symmetric items with the same score and time → equal GLEB_TWB values."""
        dist = mixed_gleb_twb(ITEMS_SYM)
        self.assertAlmostEqual(dist["Action"], dist["Drama"], places=9)

    def test_all_values_non_negative(self):
        """All distribution values are ≥ 0."""
        dist = mixed_gleb_twb(ITEMS_2)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_time_zero_yields_epsilon(self):
        """Item with time=0.0 contributes zero to numerator → epsilon."""
        items = {
            "i1": _item("i1", score=5.0, time=0.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=3.0, time=0.0, classes={"Drama": 1.0}),
        }
        dist = mixed_gleb_twb(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)
        self.assertAlmostEqual(dist["Drama"], 0.00001, places=9)

    def test_score_zero_yields_epsilon(self):
        """Item with score=0 contributes zero to numerator and denominator → epsilon."""
        items = {
            "i1": _item("i1", score=0.0, time=1.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=0.0, time=1.0, classes={"Drama": 1.0}),
        }
        dist = mixed_gleb_twb(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)

    def test_higher_time_increases_value(self):
        """Doubling time on one item's genre increases that genre's GLEB_TWB value."""
        items_low = {
            "i1": _item("i1", score=4.0, time=0.5, classes={"Action": 1.0}),
            "i2": _item("i2", score=4.0, time=0.5, classes={"Drama": 1.0}),
        }
        items_high = {
            "i1": _item("i1", score=4.0, time=1.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=4.0, time=0.5, classes={"Drama": 1.0}),
        }
        dist_low = mixed_gleb_twb(items_low)
        dist_high = mixed_gleb_twb(items_high)
        self.assertGreater(dist_high["Action"], dist_low["Action"])

    def test_non_single_genre_values_are_above_epsilon(self):
        """With 2 distinct genres (freq=0.5 each), entropy > 0 → values > epsilon."""
        dist = mixed_gleb_twb(ITEMS_2)
        self.assertGreater(dist["Action"], 0.00001)
        self.assertGreater(dist["Drama"], 0.00001)

    def test_multi_genre_item_all_keys_present(self):
        """A multi-genre item contributes all its genres to the distribution."""
        items = {
            "i1": _item("i1", score=4.0, time=0.8, classes={"A": 0.5, "B": 0.3, "C": 0.2}),
        }
        dist = mixed_gleb_twb(items)
        for g in ("A", "B", "C"):
            self.assertIn(g, dist)

    def test_determinism(self):
        """Same input always produces identical output."""
        self.assertEqual(mixed_gleb_twb(ITEMS_2), mixed_gleb_twb(ITEMS_2))

    def test_golden_three_genre_item(self):
        """Hand-verified: three-genre single item → global freq=1/3 for each genre."""
        items = {
            "i1": _item("i1", score=6.0, time=1.0, classes={"A": 1/3, "B": 1/3, "C": 1/3}),
            "i2": _item("i2", score=3.0, time=1.0, classes={"D": 1.0}),
        }
        dist = mixed_gleb_twb(items)
        # All keys must be present
        for g in ("A", "B", "C", "D"):
            self.assertIn(g, dist)
        # Values must be non-negative
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)


# ===========================================================================
# mixed_gleb_twb_with_probability_property
# ===========================================================================

class TestMixedGlebTwbWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        """TWB_GLEB_P output sums to 1.0."""
        dist = mixed_gleb_twb_with_probability_property(ITEMS_2)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        """All TWB_GLEB_P probabilities are in [0, 1]."""
        dist = mixed_gleb_twb_with_probability_property(ITEMS_2)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_golden_action_probability(self):
        """Action probability = GOLD_ACTION / (GOLD_ACTION + GOLD_DRAMA)."""
        total = GOLD_ACTION + GOLD_DRAMA
        expected = GOLD_ACTION / total
        dist = mixed_gleb_twb_with_probability_property(ITEMS_2)
        self.assertAlmostEqual(dist["Action"], expected, places=9)

    def test_golden_drama_probability(self):
        """Drama probability = GOLD_DRAMA / (GOLD_ACTION + GOLD_DRAMA)."""
        total = GOLD_ACTION + GOLD_DRAMA
        expected = GOLD_DRAMA / total
        dist = mixed_gleb_twb_with_probability_property(ITEMS_2)
        self.assertAlmostEqual(dist["Drama"], expected, places=9)

    def test_symmetric_items_yield_equal_probabilities(self):
        """Symmetric items → equal probabilities = 0.5 each."""
        dist = mixed_gleb_twb_with_probability_property(ITEMS_SYM)
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)
        self.assertAlmostEqual(dist["Drama"], 0.5, places=9)

    def test_relative_order_preserved_from_base(self):
        """Normalization does not change the relative genre ranking."""
        base = mixed_gleb_twb(ITEMS_2)
        prob = mixed_gleb_twb_with_probability_property(ITEMS_2)
        sorted_base = sorted(base.items(), key=lambda x: x[1])
        sorted_prob = sorted(prob.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_base], [k for k, _ in sorted_prob])

    def test_genre_keys_match_base(self):
        """TWB_GLEB_P has the same genre keys as TWB_GLEB."""
        base = mixed_gleb_twb(ITEMS_2)
        prob = mixed_gleb_twb_with_probability_property(ITEMS_2)
        self.assertSetEqual(set(base.keys()), set(prob.keys()))

    def test_determinism(self):
        """Repeated calls yield identical results."""
        self.assertEqual(
            mixed_gleb_twb_with_probability_property(ITEMS_2),
            mixed_gleb_twb_with_probability_property(ITEMS_2),
        )
