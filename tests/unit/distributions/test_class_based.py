"""
Unit tests for scikit_pierre.distributions.class_based.
"""
import unittest

import pytest

from scikit_pierre.distributions.class_based import (
    class_weighted_strategy,
    pure_genre,
    pure_genre_with_probability_property,
    weighted_probability_strategy,
)
from scikit_pierre.models.item import Item


# ── fixtures ───────────────────────────────────────────────────────────────────

def _item(iid, score, classes):
    return Item(_id=iid, score=score, classes=classes)


# Three-item setup replicated from the original (commented-out) test vectors.
# Items: compadecida, amor, sol; user scores: 5, 4, 4.
ITEMS_3 = {
    "compadecida": _item("compadecida", score=5.0,
                         classes={"Adventure": 0.5, "Comedy": 0.5}),
    "amor":        _item("amor",        score=4.0,
                         classes={"Drama": 1.0}),
    "sol":         _item("sol",         score=4.0,
                         classes={"Adventure": 0.25, "Crime": 0.25,
                                  "Drama": 0.25, "Western": 0.25}),
}

# CWS golden values (hand-verified, see test file header):
#   Adventure = (5*0.5 + 4*0.25) / (5 + 4) = 3.5 / 9
#   Comedy    = (5*0.5)           / 5        = 0.5
#   Drama     = (4*1.0 + 4*0.25) / (4 + 4)  = 5.0 / 8
#   Crime     = (4*0.25)          / 4        = 0.25
#   Western   = (4*0.25)          / 4        = 0.25
CWS_ADV  = 3.5 / 9
CWS_COM  = 0.5
CWS_DRA  = 5.0 / 8
CWS_CRI  = 0.25
CWS_WES  = 0.25


# ── class_weighted_strategy ────────────────────────────────────────────────────

class TestClassWeightedStrategy(unittest.TestCase):

    def test_single_item_single_genre_returns_one(self):
        """Single item with a single genre and non-zero score → CWS = 1.0."""
        items = {"i1": _item("i1", score=5.0, classes={"Action": 1.0})}
        dist = class_weighted_strategy(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_single_item_two_genres_equal_weight(self):
        """Single item with two equal-weight genres → CWS = 0.5 each."""
        items = {"i1": _item("i1", score=4.0, classes={"Action": 0.5, "Drama": 0.5})}
        dist = class_weighted_strategy(items)
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)
        self.assertAlmostEqual(dist["Drama"], 0.5, places=9)

    def test_golden_adventure_value(self):
        """CWS('Adventure') for the three-item fixture matches the hand-computed value."""
        dist = class_weighted_strategy(ITEMS_3)
        self.assertAlmostEqual(dist["Adventure"], CWS_ADV, places=9)

    def test_golden_comedy_value(self):
        """CWS('Comedy') for the three-item fixture = 0.5."""
        dist = class_weighted_strategy(ITEMS_3)
        self.assertAlmostEqual(dist["Comedy"], CWS_COM, places=9)

    def test_golden_drama_value(self):
        """CWS('Drama') for the three-item fixture = 5/8."""
        dist = class_weighted_strategy(ITEMS_3)
        self.assertAlmostEqual(dist["Drama"], CWS_DRA, places=9)

    def test_golden_crime_value(self):
        """CWS('Crime') for the three-item fixture = 0.25."""
        dist = class_weighted_strategy(ITEMS_3)
        self.assertAlmostEqual(dist["Crime"], CWS_CRI, places=9)

    def test_zero_score_item_returns_epsilon(self):
        """An item with score=0 produces the epsilon sentinel for all its genres."""
        items = {"i1": _item("i1", score=0.0, classes={"Action": 1.0})}
        dist = class_weighted_strategy(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)

    def test_genre_keys_match_item_classes(self):
        """The returned distribution has exactly the genres found in the items."""
        items = {"i1": _item("i1", score=5.0, classes={"Action": 0.6, "Comedy": 0.4})}
        dist = class_weighted_strategy(items)
        self.assertSetEqual(set(dist.keys()), {"Action", "Comedy"})

    def test_values_non_negative(self):
        """All CWS values are ≥ 0."""
        dist = class_weighted_strategy(ITEMS_3)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_returns_dict(self):
        """Return type is a dict."""
        items = {"i1": _item("i1", score=3.0, classes={"Action": 1.0})}
        self.assertIsInstance(class_weighted_strategy(items), dict)

    def test_equal_scores_higher_genre_weight_yields_higher_cws(self):
        """A genre with higher weight across equal-score items has a higher CWS."""
        items = {
            "i1": _item("i1", score=1.0, classes={"Action": 0.8, "Drama": 0.2}),
            "i2": _item("i2", score=1.0, classes={"Action": 0.8, "Drama": 0.2}),
        }
        dist = class_weighted_strategy(items)
        self.assertGreater(dist["Action"], dist["Drama"])

    def test_determinism(self):
        """Same input always produces identical output."""
        d1 = class_weighted_strategy(ITEMS_3)
        d2 = class_weighted_strategy(ITEMS_3)
        self.assertEqual(d1, d2)


# ── weighted_probability_strategy ─────────────────────────────────────────────

class TestWeightedProbabilityStrategy(unittest.TestCase):

    def test_normalizes_to_one(self):
        """WPS output sums to 1.0 (proper probability distribution)."""
        dist = weighted_probability_strategy(ITEMS_3)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_golden_values_match_original_test_vectors(self):
        """Hand-verified WPS values match the legacy test vectors (3 d.p.)."""
        dist = weighted_probability_strategy(ITEMS_3)
        total = CWS_ADV + CWS_COM + CWS_DRA + CWS_CRI + CWS_WES
        self.assertAlmostEqual(dist["Adventure"], CWS_ADV / total, places=3)
        self.assertAlmostEqual(dist["Comedy"],    CWS_COM / total, places=3)
        self.assertAlmostEqual(dist["Drama"],     CWS_DRA / total, places=3)

    def test_all_values_non_negative(self):
        """All WPS probabilities are ≥ 0."""
        dist = weighted_probability_strategy(ITEMS_3)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_all_values_at_most_one(self):
        """No WPS probability exceeds 1.0."""
        dist = weighted_probability_strategy(ITEMS_3)
        for v in dist.values():
            self.assertLessEqual(v, 1.0)

    def test_single_genre_item_probability_is_one(self):
        """Single item with single genre → WPS = 1.0 for that genre."""
        items = {"i1": _item("i1", score=5.0, classes={"Action": 1.0})}
        dist = weighted_probability_strategy(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_relative_order_preserved(self):
        """WPS preserves the relative order of genre weights from CWS."""
        cws = class_weighted_strategy(ITEMS_3)
        wps = weighted_probability_strategy(ITEMS_3)
        sorted_cws = sorted(cws.items(), key=lambda x: x[1])
        sorted_wps = sorted(wps.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_cws], [k for k, _ in sorted_wps])

    def test_determinism(self):
        """Same input always produces identical output."""
        self.assertEqual(
            weighted_probability_strategy(ITEMS_3),
            weighted_probability_strategy(ITEMS_3),
        )


# ── pure_genre ─────────────────────────────────────────────────────────────────

class TestPureGenre(unittest.TestCase):

    def test_single_item_single_genre_accumulates_weight(self):
        """Single item with genre weight 1.0 → PGD value = 1.0."""
        items = {"i1": _item("i1", score=0.0, classes={"Action": 1.0})}
        dist = pure_genre(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_two_items_same_genre_accumulates(self):
        """Two items with the same genre → weights are summed."""
        items = {
            "i1": _item("i1", score=1.0, classes={"Action": 0.5}),
            "i2": _item("i2", score=1.0, classes={"Action": 0.5}),
        }
        dist = pure_genre(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_golden_adventure_value(self):
        """PGD('Adventure') = 0.5 (compadecida) + 0.25 (sol) = 0.75."""
        dist = pure_genre(ITEMS_3)
        self.assertAlmostEqual(dist["Adventure"], 0.75, places=9)

    def test_golden_drama_value(self):
        """PGD('Drama') = 1.0 (amor) + 0.25 (sol) = 1.25."""
        dist = pure_genre(ITEMS_3)
        self.assertAlmostEqual(dist["Drama"], 1.25, places=9)

    def test_genre_keys_union_of_all_items(self):
        """Keys in PGD are the union of all genre labels across all items."""
        dist = pure_genre(ITEMS_3)
        self.assertSetEqual(
            set(dist.keys()),
            {"Adventure", "Comedy", "Drama", "Crime", "Western"},
        )

    def test_score_ignored(self):
        """PGD ignores item.score; only genre weights matter."""
        items_high = {"i1": _item("i1", score=100.0, classes={"Action": 0.5})}
        items_low  = {"i1": _item("i1", score=0.001, classes={"Action": 0.5})}
        self.assertEqual(pure_genre(items_high), pure_genre(items_low))

    def test_all_values_non_negative(self):
        """All PGD values are ≥ 0."""
        dist = pure_genre(ITEMS_3)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_determinism(self):
        """Same input produces identical output."""
        self.assertEqual(pure_genre(ITEMS_3), pure_genre(ITEMS_3))


# ── pure_genre_with_probability_property ──────────────────────────────────────

class TestPureGenreWithProbabilityProperty(unittest.TestCase):

    def test_normalizes_to_one(self):
        """PGD_P sums to 1.0 (proper probability distribution)."""
        dist = pure_genre_with_probability_property(ITEMS_3)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_golden_adventure_probability(self):
        """PGD_P('Adventure') = 0.75 / total(PGD)."""
        pgd = pure_genre(ITEMS_3)
        total = sum(pgd.values())
        dist = pure_genre_with_probability_property(ITEMS_3)
        self.assertAlmostEqual(dist["Adventure"], 0.75 / total, places=9)

    def test_golden_drama_probability(self):
        """PGD_P('Drama') = 1.25 / total(PGD)."""
        pgd = pure_genre(ITEMS_3)
        total = sum(pgd.values())
        dist = pure_genre_with_probability_property(ITEMS_3)
        self.assertAlmostEqual(dist["Drama"], 1.25 / total, places=9)

    def test_all_values_in_zero_one(self):
        """Every PGD_P probability is in [0, 1]."""
        dist = pure_genre_with_probability_property(ITEMS_3)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_single_genre_probability_is_one(self):
        """Single genre item → PGD_P = 1.0 for that genre."""
        items = {"i1": _item("i1", score=1.0, classes={"Action": 1.0})}
        dist = pure_genre_with_probability_property(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_two_equal_genres_each_probability_is_half(self):
        """Two genres with equal accumulated weight → 0.5 each."""
        items = {
            "i1": _item("i1", score=1.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=1.0, classes={"Drama": 1.0}),
        }
        dist = pure_genre_with_probability_property(items)
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)
        self.assertAlmostEqual(dist["Drama"], 0.5, places=9)

    def test_relative_order_preserved_from_pgd(self):
        """Normalizing PGD does not change the relative order of genres."""
        pgd  = pure_genre(ITEMS_3)
        pgdp = pure_genre_with_probability_property(ITEMS_3)
        sorted_pgd  = sorted(pgd.items(),  key=lambda x: x[1])
        sorted_pgdp = sorted(pgdp.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_pgd], [k for k, _ in sorted_pgdp])

    def test_determinism(self):
        """Same input always produces identical output."""
        self.assertEqual(
            pure_genre_with_probability_property(ITEMS_3),
            pure_genre_with_probability_property(ITEMS_3),
        )
