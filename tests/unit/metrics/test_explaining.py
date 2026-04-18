"""
Unit tests for scikit_pierre.metrics.explaining.ExplainingMiscalibration.

Covers: single_process_anic, compute_miscalibration, find_user_based_on_changes,
user_analyzing_genres, compute (returns 0.0), and no-raise checks for print methods.
"""
import io
import sys
import unittest

import pandas as pd

from scikit_pierre.metrics.explaining import ExplainingMiscalibration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _profile_df():
    """2-user preference history: USER_ID, ITEM_ID, TRANSACTION_VALUE."""
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 2, 2, 2],
        "ITEM_ID": [10, 20, 30, 10, 20, 40],
        "TRANSACTION_VALUE": [5.0, 4.0, 3.0, 4.0, 5.0, 3.0],
    })


def _items_df():
    """Item catalogue: ITEM_ID, GENRES."""
    return pd.DataFrame({
        "ITEM_ID": [10, 20, 30, 40, 50],
        "GENRES": ["Action|Comedy", "Drama", "Action", "Comedy|Drama", "Thriller"],
    })


def _rec_df():
    """Calibrated recommendation lists: USER_ID, ITEM_ID, ORDER, TRANSACTION_VALUE."""
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 2, 2, 2],
        "ITEM_ID": [10, 30, 50, 10, 40, 20],
        "ORDER": [1, 2, 3, 1, 2, 3],
        "TRANSACTION_VALUE": [5.0, 3.0, 1.0, 4.0, 3.0, 5.0],
    })


def _baseline_df():
    """Baseline recommendation lists: USER_ID, ITEM_ID, ORDER, TRANSACTION_VALUE."""
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 2, 2, 2],
        "ITEM_ID": [20, 30, 10, 20, 10, 40],
        "ORDER": [1, 2, 3, 1, 2, 3],
        "TRANSACTION_VALUE": [4.0, 3.0, 5.0, 5.0, 4.0, 3.0],
    })


def _make_metric(dist="CWS", measure="KL"):
    return ExplainingMiscalibration(
        users_profile_df=_profile_df(),
        users_rec_list_df=_rec_df(),
        users_baseline_df=_baseline_df(),
        items_df=_items_df(),
        distribution_name=dist,
        distance_func_name=measure,
    )


# ---------------------------------------------------------------------------
# single_process_anic (static)
# ---------------------------------------------------------------------------

class TestSingleProcessAnic(unittest.TestCase):

    def _group(self, uid, item_ids):
        return (uid, pd.DataFrame({"USER_ID": [uid] * len(item_ids), "ITEM_ID": item_ids}))

    def test_no_change_returns_zero(self):
        """Identical item sets → 0 new items in rec."""
        g2 = self._group(1, [10, 20, 30])
        g3 = self._group(1, [10, 20, 30])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 0)

    def test_complete_change_returns_list_size(self):
        """Fully disjoint item sets → all items are new."""
        g2 = self._group(1, [40, 50, 60])
        g3 = self._group(1, [10, 20, 30])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 3)

    def test_partial_change(self):
        """One item replaced → ANIC = 1."""
        g2 = self._group(1, [10, 20, 50])
        g3 = self._group(1, [10, 20, 30])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 1)

    def test_rec_empty_returns_zero(self):
        """Empty rec list → 0 new items (empty set minus anything = empty set)."""
        g2 = self._group(1, [])
        g3 = self._group(1, [10, 20])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 0)

    def test_baseline_empty_all_new(self):
        """Baseline empty → all rec items are new."""
        g2 = self._group(1, [10, 20])
        g3 = self._group(1, [])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 2)

    def test_returns_int(self):
        """Return type is an int (or int-like)."""
        g2 = self._group(1, [10, 20, 30])
        g3 = self._group(1, [10, 20, 30])
        result = ExplainingMiscalibration.single_process_anic(g2, g3)
        self.assertIsInstance(result, int)

    def test_order_invariant(self):
        """Set difference is order-invariant."""
        g2 = self._group(1, [10, 20, 30])
        g3 = self._group(1, [30, 20, 10])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 0)

    def test_duplicate_items_counted_once(self):
        """Duplicates in the item lists collapse via set() → counted once."""
        g2 = self._group(1, [10, 10, 50])
        g3 = self._group(1, [10, 20, 30])
        self.assertEqual(ExplainingMiscalibration.single_process_anic(g2, g3), 1)


# ---------------------------------------------------------------------------
# user_analyzing_genres (static, prints)
# ---------------------------------------------------------------------------

class TestUserAnalyzingGenres(unittest.TestCase):

    def test_does_not_raise_with_disjoint_genres(self):
        """Disjoint genre lists must not raise."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            ExplainingMiscalibration.user_analyzing_genres(["Action"], ["Drama"])
        finally:
            sys.stdout = sys.__stdout__

    def test_does_not_raise_with_empty_genres(self):
        """Empty genre lists must not raise."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            ExplainingMiscalibration.user_analyzing_genres([], [])
        finally:
            sys.stdout = sys.__stdout__

    def test_does_not_raise_with_overlap(self):
        """Overlapping genre lists must not raise."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            ExplainingMiscalibration.user_analyzing_genres(["Action", "Drama"], ["Drama", "Comedy"])
        finally:
            sys.stdout = sys.__stdout__


# ---------------------------------------------------------------------------
# compute_miscalibration
# ---------------------------------------------------------------------------

class TestComputeMiscalibration(unittest.TestCase):

    def setUp(self):
        self.metric = _make_metric()
        # Prime items in memory (needed for internal distribution lookups).
        self.metric.item_preparation()
        self.metric.compute_target_dist()

    def test_identical_distributions_non_negative(self):
        """Identical p and q → divergence ≥ 0 (after tilde-q smoothing KL ≈ 0)."""
        dist = {"Action": 0.5, "Drama": 0.5}
        result = self.metric.compute_miscalibration(dist, dist)
        self.assertGreaterEqual(result, 0.0)

    def test_returns_float(self):
        """compute_miscalibration returns a float."""
        dist = {"Action": 0.6, "Drama": 0.4}
        result = self.metric.compute_miscalibration(dist, dist)
        self.assertIsInstance(result, float)

    def test_diverge_distributions_non_negative(self):
        """Divergent distributions produce a non-negative divergence."""
        p = {"Action": 0.8, "Drama": 0.2}
        q = {"Action": 0.2, "Drama": 0.8}
        result = self.metric.compute_miscalibration(p, q)
        self.assertGreaterEqual(result, 0.0)

    def test_finite_result(self):
        """Result is always finite (no NaN/inf)."""
        import math
        p = {"Action": 0.7, "Drama": 0.3}
        q = {"Action": 0.3, "Drama": 0.7}
        result = self.metric.compute_miscalibration(p, q)
        self.assertTrue(math.isfinite(result))

    def test_deterministic(self):
        """Same distributions always produce the same divergence."""
        p = {"Action": 0.6, "Drama": 0.4}
        q = {"Action": 0.4, "Drama": 0.6}
        r1 = self.metric.compute_miscalibration(p, q)
        r2 = self.metric.compute_miscalibration(p, q)
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# find_user_based_on_changes
# ---------------------------------------------------------------------------

class TestFindUserBasedOnChanges(unittest.TestCase):

    def setUp(self):
        self.metric = _make_metric()
        self.metric.item_preparation()
        self.metric.compute_target_dist()
        self.metric.ordering_and_grouping()

    def test_returns_dict(self):
        """find_user_based_on_changes returns a dict."""
        result = self.metric.find_user_based_on_changes()
        self.assertIsInstance(result, dict)

    def test_keys_are_string_user_ids(self):
        """Keys are string representations of user IDs."""
        result = self.metric.find_user_based_on_changes()
        for k in result.keys():
            self.assertIsInstance(k, str)

    def test_all_users_present(self):
        """All users from rec_df appear in the result."""
        result = self.metric.find_user_based_on_changes()
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_values_are_non_negative_integers(self):
        """ANIC values are non-negative integers."""
        result = self.metric.find_user_based_on_changes()
        for v in result.values():
            self.assertGreaterEqual(v, 0)
            self.assertIsInstance(v, int)

    def test_completely_same_lists_yield_zero(self):
        """When rec and baseline have the same items, ANIC = 0 for all users."""
        same_rec = _baseline_df().copy()
        metric = ExplainingMiscalibration(
            users_profile_df=_profile_df(),
            users_rec_list_df=same_rec,
            users_baseline_df=_baseline_df(),
            items_df=_items_df(),
        )
        metric.item_preparation()
        metric.compute_target_dist()
        metric.ordering_and_grouping()
        result = metric.find_user_based_on_changes()
        for v in result.values():
            self.assertEqual(v, 0)


# ---------------------------------------------------------------------------
# compute() — full pipeline
# ---------------------------------------------------------------------------

class TestExplainingMiscalibrationCompute(unittest.TestCase):

    def test_compute_returns_zero(self):
        """compute() always returns the sentinel 0.0."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            result = _make_metric().compute()
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(result, 0.0)

    def test_compute_kl_does_not_raise(self):
        """Full compute() with KL fairness must not raise."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            _make_metric(dist="CWS", measure="KL").compute()
        finally:
            sys.stdout = sys.__stdout__

    def test_compute_hellinger_does_not_raise(self):
        """Full compute() with HELLINGER fairness must not raise."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            _make_metric(dist="CWS", measure="HELLINGER").compute()
        finally:
            sys.stdout = sys.__stdout__

    def test_compute_deterministic(self):
        """Calling compute() twice on identical inputs returns 0.0 both times."""
        for _ in range(2):
            captured = io.StringIO()
            sys.stdout = captured
            try:
                result = _make_metric().compute()
            finally:
                sys.stdout = sys.__stdout__
            self.assertEqual(result, 0.0)

    def test_compute_integer_user_ids_does_not_raise(self):
        """Regression: compute() must not KeyError when USER_ID is integer type."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            result = _make_metric().compute()
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(result, 0.0)
