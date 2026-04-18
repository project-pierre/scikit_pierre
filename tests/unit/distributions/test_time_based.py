"""
Unit tests for scikit_pierre.distributions.time_based.
"""
import unittest

from scikit_pierre.distributions.time_based import (
    time_genre,
    time_genre_with_probability_property,
    time_weighted_based,
    time_weighted_based_with_probability_property,
)
from scikit_pierre.models.item import Item


def _item(iid, score, time, classes):
    return Item(_id=iid, score=score, time=time, classes=classes)


# ── time_weighted_based ────────────────────────────────────────────────────────

class TestTimeWeightedBased(unittest.TestCase):

    def test_single_item_returns_genre_weight(self):
        """
        Single item: TWB(g) = time*score*weight / score = time*weight.
        With time=1.0, weight=1.0 → TWB = 1.0.
        """
        items = {"i1": _item("i1", score=5.0, time=1.0, classes={"Action": 1.0})}
        dist = time_weighted_based(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_time_zero_yields_epsilon(self):
        """Item with time=0.0 contributes zero to the numerator → epsilon returned."""
        items = {"i1": _item("i1", score=5.0, time=0.0, classes={"Action": 1.0})}
        dist = time_weighted_based(items)
        self.assertAlmostEqual(dist["Action"], 0.00001, places=9)

    def test_recent_item_outweighs_old_item(self):
        """
        Two items with the same genre and score but different times:
        the distribution value reflects a higher contribution from the recent one.
        """
        items = {
            "new": _item("new", score=4.0, time=1.0, classes={"Action": 1.0}),
            "old": _item("old", score=4.0, time=0.0, classes={"Action": 1.0}),
        }
        dist = time_weighted_based(items)
        # numerator = 1.0*4*1 + 0.0*4*1 = 4; denominator = 4+4 = 8
        # TWB = 4/8 = 0.5
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)

    def test_equal_time_reduces_to_time_times_weight(self):
        """
        When all items share the same time t and genre weight w,
        TWB(g) = (t*score*w + t*score*w) / (score + score) = t * w.
        With t=0.5, w=1.0 → TWB = 0.5.
        """
        t = 0.5
        items = {
            "i1": _item("i1", score=4.0, time=t, classes={"Action": 1.0}),
            "i2": _item("i2", score=4.0, time=t, classes={"Action": 1.0}),
        }
        dist = time_weighted_based(items)
        # numerator = 2*(t*4*1) = 4.0; denominator = 8 → TWB = 0.5 = t*1.0
        self.assertAlmostEqual(dist["Action"], t * 1.0, places=9)

    def test_golden_two_genres_different_times(self):
        """Hand-verified golden values for two items with different times and genres."""
        items = {
            "i1": _item("i1", score=5.0, time=1.0, classes={"Action": 0.5, "Drama": 0.5}),
            "i2": _item("i2", score=3.0, time=0.5, classes={"Action": 0.5, "Drama": 0.5}),
        }
        # numerator['Action'] = 1.0*5*0.5 + 0.5*3*0.5 = 2.5 + 0.75 = 3.25
        # denominator['Action'] = 5 + 3 = 8
        # TWB('Action') = 3.25/8 = 0.40625
        dist = time_weighted_based(items)
        self.assertAlmostEqual(dist["Action"], 3.25 / 8, places=9)
        self.assertAlmostEqual(dist["Drama"],  3.25 / 8, places=9)

    def test_all_values_non_negative(self):
        """All TWB values are ≥ 0."""
        items = {
            "i1": _item("i1", score=5.0, time=0.8, classes={"Action": 0.6, "Drama": 0.4}),
            "i2": _item("i2", score=3.0, time=0.2, classes={"Action": 0.3, "Sci-Fi": 0.7}),
        }
        dist = time_weighted_based(items)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_returns_dict(self):
        """Return type is a dict."""
        items = {"i1": _item("i1", score=3.0, time=0.5, classes={"Action": 1.0})}
        self.assertIsInstance(time_weighted_based(items), dict)

    def test_determinism(self):
        """Same input always produces identical output."""
        items = {"i1": _item("i1", score=4.0, time=0.7, classes={"Action": 1.0})}
        self.assertEqual(time_weighted_based(items), time_weighted_based(items))


# ── time_weighted_based_with_probability_property ─────────────────────────────

class TestTimeWeightedBasedWithProbabilityProperty(unittest.TestCase):

    def _items(self):
        return {
            "i1": _item("i1", score=5.0, time=0.9, classes={"Action": 0.5, "Drama": 0.5}),
            "i2": _item("i2", score=3.0, time=0.4, classes={"Action": 0.5, "Comedy": 0.5}),
        }

    def test_normalizes_to_one(self):
        """TWB_P output sums to 1.0."""
        dist = time_weighted_based_with_probability_property(self._items())
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        """All TWB_P probabilities are in [0, 1]."""
        dist = time_weighted_based_with_probability_property(self._items())
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_single_genre_item_probability_is_one(self):
        """Single item, single genre → TWB_P = 1.0."""
        items = {"i1": _item("i1", score=5.0, time=0.5, classes={"Action": 1.0})}
        dist = time_weighted_based_with_probability_property(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_relative_order_preserved_from_twb(self):
        """Normalization does not change the relative genre ranking."""
        items = self._items()
        twb  = time_weighted_based(items)
        twbp = time_weighted_based_with_probability_property(items)
        sorted_twb  = sorted(twb.items(),  key=lambda x: x[1])
        sorted_twbp = sorted(twbp.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_twb], [k for k, _ in sorted_twbp])

    def test_determinism(self):
        """Repeated calls yield identical results."""
        items = self._items()
        self.assertEqual(
            time_weighted_based_with_probability_property(items),
            time_weighted_based_with_probability_property(items),
        )


# ── time_genre ─────────────────────────────────────────────────────────────────

class TestTimeGenre(unittest.TestCase):

    def test_single_item_single_genre_returns_one(self):
        """
        TGD with one item: numerator = time*weight, denominator = time
        → TGD = weight. With weight=1.0, returns 1.0.
        """
        items = {"i1": _item("i1", score=0.0, time=1.0, classes={"Action": 1.0})}
        dist = time_genre(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_single_item_two_equal_genres(self):
        """Single item with equal genre weights → TGD = 0.5 each."""
        items = {"i1": _item("i1", score=0.0, time=1.0, classes={"Action": 0.5, "Drama": 0.5})}
        dist = time_genre(items)
        self.assertAlmostEqual(dist["Action"], 0.5, places=9)
        self.assertAlmostEqual(dist["Drama"],  0.5, places=9)

    def test_score_does_not_affect_tgd(self):
        """TGD ignores item.score; only time and genre weight matter."""
        items_high = {"i1": _item("i1", score=100.0, time=1.0, classes={"Action": 1.0})}
        items_low  = {"i1": _item("i1", score=0.001, time=1.0, classes={"Action": 1.0})}
        self.assertAlmostEqual(
            time_genre(items_high)["Action"],
            time_genre(items_low)["Action"],
            places=9,
        )

    def test_two_items_same_genre_equal_times_returns_weight(self):
        """
        Two items, same genre, equal times:
        numerator = t*w1 + t*w2, denominator = t + t
        → TGD = (w1+w2)/2 = 0.5 for w1=w2=0.5.
        Wait: each item has Action=1.0, so TGD = (t*1 + t*1)/(t+t) = 1.0.
        """
        items = {
            "i1": _item("i1", score=3.0, time=0.5, classes={"Action": 1.0}),
            "i2": _item("i2", score=5.0, time=0.5, classes={"Action": 1.0}),
        }
        dist = time_genre(items)
        # num = 0.5 + 0.5 = 1.0; den = 0.5 + 0.5 = 1.0 → TGD = 1.0
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_recent_item_shifts_distribution(self):
        """
        Recent item (time=1) with Action and old item (time=0) with Drama:
        only Action gets numerator > 0; Drama gets epsilon.
        """
        items = {
            "i1": _item("i1", score=3.0, time=1.0, classes={"Action": 1.0}),
            "i2": _item("i2", score=3.0, time=0.0, classes={"Drama": 1.0}),
        }
        dist = time_genre(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)
        self.assertAlmostEqual(dist["Drama"],  0.00001, places=9)

    def test_golden_value_two_items_two_times(self):
        """Hand-verified: two items with different times sharing one genre."""
        items = {
            "i1": _item("i1", score=0.0, time=0.8, classes={"Action": 1.0}),
            "i2": _item("i2", score=0.0, time=0.2, classes={"Action": 1.0}),
        }
        # num = 0.8 + 0.2 = 1.0; den = 0.8 + 0.2 = 1.0 → TGD = 1.0
        dist = time_genre(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_all_values_non_negative(self):
        """All TGD values are ≥ 0."""
        items = {
            "i1": _item("i1", score=3.0, time=0.6, classes={"Action": 0.5, "Drama": 0.5}),
            "i2": _item("i2", score=2.0, time=0.3, classes={"Comedy": 1.0}),
        }
        dist = time_genre(items)
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)

    def test_returns_dict(self):
        """Return type is a dict."""
        items = {"i1": _item("i1", score=1.0, time=0.5, classes={"Action": 1.0})}
        self.assertIsInstance(time_genre(items), dict)

    def test_determinism(self):
        """Same input produces identical output."""
        items = {"i1": _item("i1", score=3.0, time=0.7, classes={"Action": 1.0})}
        self.assertEqual(time_genre(items), time_genre(items))


# ── time_genre_with_probability_property ──────────────────────────────────────

class TestTimeGenreWithProbabilityProperty(unittest.TestCase):

    def _items(self):
        return {
            "i1": _item("i1", score=4.0, time=0.8, classes={"Action": 0.5, "Drama": 0.5}),
            "i2": _item("i2", score=2.0, time=0.3, classes={"Comedy": 1.0}),
        }

    def test_normalizes_to_one(self):
        """TGD_P output sums to 1.0."""
        dist = time_genre_with_probability_property(self._items())
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_all_values_in_zero_one(self):
        """All TGD_P probabilities are in [0, 1]."""
        dist = time_genre_with_probability_property(self._items())
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_single_genre_probability_is_one(self):
        """Single genre → TGD_P = 1.0 for that genre."""
        items = {"i1": _item("i1", score=0.0, time=0.5, classes={"Action": 1.0})}
        dist = time_genre_with_probability_property(items)
        self.assertAlmostEqual(dist["Action"], 1.0, places=9)

    def test_relative_order_preserved_from_tgd(self):
        """Normalization does not change the relative genre ranking."""
        items = self._items()
        tgd  = time_genre(items)
        tgdp = time_genre_with_probability_property(items)
        sorted_tgd  = sorted(tgd.items(),  key=lambda x: x[1])
        sorted_tgdp = sorted(tgdp.items(), key=lambda x: x[1])
        self.assertEqual([k for k, _ in sorted_tgd], [k for k, _ in sorted_tgdp])

    def test_determinism(self):
        """Repeated calls produce identical results."""
        items = self._items()
        self.assertEqual(
            time_genre_with_probability_property(items),
            time_genre_with_probability_property(items),
        )
