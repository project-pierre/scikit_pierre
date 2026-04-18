"""
Unit tests for scikit_pierre.metrics.evaluation.
"""
import math
import unittest

import numpy as np
import pandas as pd
import pytest

from scikit_pierre.metrics.evaluation import (
    AverageNumberOfGenreChanges,
    AverageNumberOfOItemsChanges,
    Coverage,
    IntraListSimilarity,
    MeanAveragePrecision,
    MeanReciprocalRank,
    Novelty,
    Personalization,
    Serendipity,
    Unexpectedness,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _rec_df(*user_item_pairs, with_order=False):
    """Build a USER_ID/ITEM_ID (and optional ORDER) DataFrame."""
    rows = []
    for uid, items in user_item_pairs:
        for rank, iid in enumerate(items, start=1):
            row = {"USER_ID": uid, "ITEM_ID": iid}
            if with_order:
                row["ORDER"] = rank
            rows.append(row)
    return pd.DataFrame(rows)


def _items_df(item_genres: dict) -> pd.DataFrame:
    """Build an ITEM_ID/GENRES DataFrame. genres are pipe-separated strings."""
    return pd.DataFrame([
        {"ITEM_ID": iid, "GENRES": genres}
        for iid, genres in item_genres.items()
    ])


# ── MeanAveragePrecision.get_list_precision ────────────────────────────────────

class TestGetListPrecision(unittest.TestCase):

    def test_get_list_precision_empty_list_returns_zero(self):
        """Empty relevance array yields 0.0."""
        self.assertEqual(MeanAveragePrecision.get_list_precision([]), 0.0)

    def test_get_list_precision_all_relevant_returns_one(self):
        """Perfect recall across all positions yields 1.0."""
        self.assertAlmostEqual(MeanAveragePrecision.get_list_precision([True, True, True]), 1.0)

    def test_get_list_precision_all_irrelevant_returns_zero(self):
        """No relevant items yield 0.0."""
        self.assertAlmostEqual(MeanAveragePrecision.get_list_precision([False, False, False]), 0.0)

    def test_get_list_precision_single_relevant(self):
        """Single True at position 1 → AP = 1.0."""
        self.assertAlmostEqual(MeanAveragePrecision.get_list_precision([True]), 1.0)

    def test_get_list_precision_single_irrelevant(self):
        """Single False at position 1 → AP = 0.0."""
        self.assertAlmostEqual(MeanAveragePrecision.get_list_precision([False]), 0.0)

    def test_get_list_precision_relevant_first_then_irrelevant(self):
        """[True, False, False]: hits = 1/1, 1/2, 1/3 → mean ≈ 0.611."""
        expected = (1.0 + 0.5 + 1 / 3) / 3
        self.assertAlmostEqual(
            MeanAveragePrecision.get_list_precision([True, False, False]),
            expected,
        )

    def test_get_list_precision_irrelevant_first_relevant_last(self):
        """[False, False, True]: hits = 0/1, 0/2, 1/3 → mean ≈ 0.111."""
        expected = (0.0 + 0.0 + 1 / 3) / 3
        self.assertAlmostEqual(
            MeanAveragePrecision.get_list_precision([False, False, True]),
            expected,
        )

    def test_get_list_precision_alternating_true_false_true(self):
        """[True, False, True]: mean of [1, 1/2, 2/3] ≈ 0.722."""
        expected = (1.0 + 0.5 + 2 / 3) / 3
        self.assertAlmostEqual(
            MeanAveragePrecision.get_list_precision([True, False, True]),
            expected,
        )

    def test_get_list_precision_result_in_zero_one_range(self):
        """AP must always be in [0, 1]."""
        for relevance in [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ]:
            ap = MeanAveragePrecision.get_list_precision(relevance)
            self.assertGreaterEqual(ap, 0.0)
            self.assertLessEqual(ap, 1.0)

    def test_get_list_precision_order_matters(self):
        """Putting the relevant item earlier increases AP."""
        ap_early = MeanAveragePrecision.get_list_precision([True, False, False])
        ap_late = MeanAveragePrecision.get_list_precision([False, False, True])
        self.assertGreater(ap_early, ap_late)

    def test_get_list_precision_two_items_first_relevant(self):
        """[True, False]: AP = mean([1/1, 1/2]) = 0.75."""
        self.assertAlmostEqual(
            MeanAveragePrecision.get_list_precision([True, False]),
            0.75,
        )


# ── MeanAveragePrecision (integration) ────────────────────────────────────────

class TestMeanAveragePrecisionCompute(unittest.TestCase):

    def _build(self, rec_pairs, test_pairs):
        rec_df = _rec_df(*rec_pairs)
        test_df = _rec_df(*test_pairs)
        return MeanAveragePrecision(users_rec_list_df=rec_df, users_test_set_df=test_df)

    def test_compute_perfect_recall_returns_one(self):
        """All recommended items are in the test set → MAP = 1.0."""
        m = self._build(
            [(1, [10, 20, 30])],
            [(1, [10, 20, 30])],
        )
        self.assertAlmostEqual(m.compute(), 1.0)

    def test_compute_no_relevant_items_returns_zero(self):
        """No overlap between rec and test → MAP = 0.0."""
        m = self._build(
            [(1, [10, 20, 30])],
            [(1, [40, 50, 60])],
        )
        self.assertAlmostEqual(m.compute(), 0.0)

    def test_compute_mean_across_users(self):
        """MAP is averaged over all users."""
        m = self._build(
            [(1, [10]), (2, [20])],
            [(1, [10]), (2, [99])],
        )
        # User 1: AP = 1.0; User 2: AP = 0.0 → MAP = 0.5
        self.assertAlmostEqual(m.compute(), 0.5)

    def test_compute_result_in_zero_one_range(self):
        """MAP must be in [0, 1]."""
        m = self._build(
            [(1, [10, 20, 30]), (2, [40, 50, 60])],
            [(1, [10, 30]), (2, [40])],
        )
        result = m.compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_compute_raises_on_user_mismatch(self):
        """Mismatched user sets raise IndexError."""
        rec_df = _rec_df((1, [10]), (2, [20]))
        test_df = _rec_df((1, [10]), (3, [30]))
        m = MeanAveragePrecision(users_rec_list_df=rec_df, users_test_set_df=test_df)
        with self.assertRaises(IndexError):
            m.compute()


# ── MeanReciprocalRank.get_list_reciprocal ─────────────────────────────────────

class TestGetListReciprocal(unittest.TestCase):

    def test_get_list_reciprocal_empty_list_returns_zero(self):
        """Empty array yields 0.0."""
        self.assertEqual(MeanReciprocalRank.get_list_reciprocal([]), 0.0)

    def test_get_list_reciprocal_first_position_relevant(self):
        """First hit at position 1 → RR = 1.0."""
        self.assertAlmostEqual(MeanReciprocalRank.get_list_reciprocal([True, False, False]), 1.0)

    def test_get_list_reciprocal_second_position_relevant(self):
        """First hit at position 2 → RR = 0.5."""
        self.assertAlmostEqual(MeanReciprocalRank.get_list_reciprocal([False, True, False]), 0.5)

    def test_get_list_reciprocal_third_position_relevant(self):
        """First hit at position 3 → RR = 1/3."""
        self.assertAlmostEqual(
            MeanReciprocalRank.get_list_reciprocal([False, False, True]),
            1 / 3,
        )

    def test_get_list_reciprocal_no_relevant_returns_zero(self):
        """No hit yields 0.0."""
        self.assertAlmostEqual(MeanReciprocalRank.get_list_reciprocal([False, False, False]), 0.0)

    def test_get_list_reciprocal_returns_first_hit_only(self):
        """Only the first relevant position counts; later hits are ignored."""
        # First hit at position 1 regardless of what follows
        rr_early = MeanReciprocalRank.get_list_reciprocal([True, True, True])
        self.assertAlmostEqual(rr_early, 1.0)

    def test_get_list_reciprocal_single_true(self):
        """Single True at position 1 → RR = 1.0."""
        self.assertAlmostEqual(MeanReciprocalRank.get_list_reciprocal([True]), 1.0)

    def test_get_list_reciprocal_single_false(self):
        """Single False → RR = 0.0."""
        self.assertAlmostEqual(MeanReciprocalRank.get_list_reciprocal([False]), 0.0)

    def test_get_list_reciprocal_result_in_zero_one_range(self):
        """RR is always in [0, 1]."""
        for relevance in [[True, False], [False, True], [False, False], [True]]:
            rr = MeanReciprocalRank.get_list_reciprocal(relevance)
            self.assertGreaterEqual(rr, 0.0)
            self.assertLessEqual(rr, 1.0)

    def test_get_list_reciprocal_earlier_hit_yields_higher_value(self):
        """A hit at position 1 always beats a hit at position 2."""
        rr_pos1 = MeanReciprocalRank.get_list_reciprocal([True, False])
        rr_pos2 = MeanReciprocalRank.get_list_reciprocal([False, True])
        self.assertGreater(rr_pos1, rr_pos2)

    def test_get_list_reciprocal_monotone_in_hit_position(self):
        """RR decreases as the hit position moves further down the list."""
        rrs = [
            MeanReciprocalRank.get_list_reciprocal([False] * k + [True] + [False])
            for k in range(5)
        ]
        for a, b in zip(rrs, rrs[1:]):
            self.assertGreater(a, b)


# ── MeanReciprocalRank (integration) ──────────────────────────────────────────

class TestMeanReciprocalRankCompute(unittest.TestCase):

    def _build(self, rec_pairs, test_pairs):
        return MeanReciprocalRank(
            users_rec_list_df=_rec_df(*rec_pairs),
            users_test_set_df=_rec_df(*test_pairs),
        )

    def test_compute_first_item_relevant_returns_one(self):
        """First recommended item is relevant → MRR = 1.0."""
        m = self._build([(1, [10, 20, 30])], [(1, [10, 99])])
        self.assertAlmostEqual(m.compute(), 1.0)

    def test_compute_no_relevant_items_returns_zero(self):
        """No overlap → MRR = 0.0."""
        m = self._build([(1, [10, 20])], [(1, [99, 98])])
        self.assertAlmostEqual(m.compute(), 0.0)

    def test_compute_averaged_across_users(self):
        """MRR is the mean of per-user RR values."""
        m = self._build(
            [(1, [10, 20]), (2, [30, 40])],
            [(1, [10]),     (2, [40])],
        )
        # User 1: RR = 1.0; User 2: first hit at pos 2 → RR = 0.5
        self.assertAlmostEqual(m.compute(), 0.75)

    def test_compute_result_in_zero_one_range(self):
        """MRR must be in [0, 1]."""
        m = self._build(
            [(1, [10, 20, 30]), (2, [40, 50, 60])],
            [(1, [30]),         (2, [40])],
        )
        result = m.compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ── Personalization ────────────────────────────────────────────────────────────

class TestPersonalization(unittest.TestCase):

    def test_compute_identical_rec_lists_returns_zero(self):
        """Users sharing the exact same list → personalization = 0.0."""
        rec_df = _rec_df((1, [10, 20, 30]), (2, [10, 20, 30]))
        result = Personalization(users_rec_list_df=rec_df).compute()
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_compute_disjoint_rec_lists_returns_one(self):
        """Users sharing no items → personalization = 1.0."""
        rec_df = _rec_df((1, [10, 20]), (2, [30, 40]))
        result = Personalization(users_rec_list_df=rec_df).compute()
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_compute_result_in_zero_one_range(self):
        """Personalization is always in [0, 1]."""
        rec_df = _rec_df((1, [10, 20, 30]), (2, [10, 40, 50]), (3, [60, 70, 80]))
        result = Personalization(users_rec_list_df=rec_df).compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_compute_partial_overlap_between_zero_and_one(self):
        """Partial item overlap yields personalization strictly between 0 and 1."""
        rec_df = _rec_df((1, [10, 20, 30]), (2, [10, 40, 50]))
        result = Personalization(users_rec_list_df=rec_df).compute()
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_compute_more_overlap_yields_lower_personalization(self):
        """Higher item overlap across users decreases personalization."""
        high_overlap = _rec_df((1, [10, 20, 30, 40]), (2, [10, 20, 30, 99]))
        low_overlap = _rec_df((1, [10, 20, 30, 40]), (2, [50, 60, 70, 80]))
        result_high = Personalization(users_rec_list_df=high_overlap).compute()
        result_low = Personalization(users_rec_list_df=low_overlap).compute()
        self.assertLess(result_high, result_low)


# ── Coverage ───────────────────────────────────────────────────────────────────

class TestCoverage(unittest.TestCase):

    def test_compute_full_catalog_coverage_returns_100(self):
        """All catalog items appear in rec lists → 100.0 %."""
        rec_df = _rec_df((1, [10, 20]), (2, [30, 40]))
        items = _items_df({10: "Action", 20: "Drama", 30: "Action", 40: "Drama"})
        result = Coverage(users_rec_list_df=rec_df, items_df=items).compute()
        self.assertAlmostEqual(result, 100.0)

    def test_compute_half_catalog_coverage_returns_50(self):
        """Half the catalog items are recommended → 50.0 %."""
        rec_df = _rec_df((1, [10, 20]))
        items = _items_df({10: "Action", 20: "Drama", 30: "Action", 40: "Drama"})
        result = Coverage(users_rec_list_df=rec_df, items_df=items).compute()
        self.assertAlmostEqual(result, 50.0)

    def test_compute_result_is_percentage(self):
        """Result is expressed as a percentage (0 – 100)."""
        rec_df = _rec_df((1, [10]))
        items = _items_df({10: "Action", 20: "Drama"})
        result = Coverage(users_rec_list_df=rec_df, items_df=items).compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)

    def test_compute_duplicates_across_users_not_double_counted(self):
        """Items recommended to multiple users are counted once."""
        rec_df = _rec_df((1, [10, 20]), (2, [10, 20]))
        items = _items_df({10: "Action", 20: "Drama", 30: "Comedy"})
        result = Coverage(users_rec_list_df=rec_df, items_df=items).compute()
        # Only 2 unique out of 3 catalog items → 66.67 %
        self.assertAlmostEqual(result, round(2 / 3 * 100, 2))

    def test_compute_single_user_single_item_from_large_catalog(self):
        """Coverage of 1 item from a catalog of 10 → 10.0 %."""
        rec_df = _rec_df((1, [10]))
        items = _items_df({i: "Action" for i in range(10, 20)})
        result = Coverage(users_rec_list_df=rec_df, items_df=items).compute()
        self.assertAlmostEqual(result, 10.0)

    def test_compute_more_items_recommended_increases_coverage(self):
        """Recommending more distinct items from the catalog increases coverage."""
        items = _items_df({i: "Action" for i in range(1, 11)})
        rec_small = _rec_df((1, [1, 2]))
        rec_large = _rec_df((1, [1, 2, 3, 4, 5]))
        cov_small = Coverage(users_rec_list_df=rec_small, items_df=items).compute()
        cov_large = Coverage(users_rec_list_df=rec_large, items_df=items).compute()
        self.assertLess(cov_small, cov_large)


# ── Novelty.single_process_nov ─────────────────────────────────────────────────

class TestSingleProcessNov(unittest.TestCase):
    """Tests for the static Novelty.single_process_nov helper."""

    def test_single_item_known_popularity(self):
        """Self-information of one item with known pop and u."""
        # -log2(pop/u) = -log2(1/10) = log2(10)
        result = Novelty.single_process_nov(
            predicted=[["1"]], pop={"1": 1}, u=10, n=1
        )
        self.assertAlmostEqual(result, math.log2(10), places=5)

    def test_all_users_see_popular_item_novelty_is_zero(self):
        """Item consumed by every user has novelty 0 (-log2(u/u) = 0)."""
        u = 5
        result = Novelty.single_process_nov(
            predicted=[["1"]], pop={"1": u}, u=u, n=1
        )
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_novelty_non_negative(self):
        """Novelty is always ≥ 0."""
        result = Novelty.single_process_nov(
            predicted=[["1", "2"]], pop={"1": 3, "2": 5}, u=10, n=2
        )
        self.assertGreaterEqual(result, 0.0)

    def test_rarer_items_yield_higher_novelty(self):
        """An item with lower popularity contributes higher self-information."""
        u = 100
        nov_rare = Novelty.single_process_nov(
            predicted=[["1"]], pop={"1": 1}, u=u, n=1
        )
        nov_common = Novelty.single_process_nov(
            predicted=[["1"]], pop={"1": 50}, u=u, n=1
        )
        self.assertGreater(nov_rare, nov_common)

    def test_multiple_users_averaged(self):
        """Novelty is the mean across all users."""
        u = 10
        # Two users: one with item pop=1 (high novelty), one with pop=10 (zero novelty)
        nov = Novelty.single_process_nov(
            predicted=[["a"], ["b"]],
            pop={"a": 1, "b": u},
            u=u,
            n=1,
        )
        expected = (math.log2(u / 1) + 0.0) / 2
        self.assertAlmostEqual(nov, expected, places=5)

    def test_zero_popularity_uses_epsilon(self):
        """Items with zero popularity use the 0.00001 epsilon, not 0."""
        u = 10
        epsilon = 0.00001
        result = Novelty.single_process_nov(
            predicted=[["x"]], pop={"x": 0}, u=u, n=1
        )
        expected = -math.log2(epsilon / u)
        self.assertAlmostEqual(result, expected, places=4)


# ── Novelty (integration) ──────────────────────────────────────────────────────

class TestNoveltyCompute(unittest.TestCase):

    def test_compute_returns_non_negative(self):
        """Novelty is always ≥ 0."""
        profile = pd.DataFrame({
            "USER_ID": ["u1", "u1", "u2"],
            "ITEM_ID": ["i1", "i2", "i1"],
        })
        rec = pd.DataFrame({
            "USER_ID": ["u1", "u1", "u2", "u2"],
            "ITEM_ID": ["i1", "i3", "i2", "i3"],
            "ORDER": [1, 2, 1, 2],
        })
        items = pd.DataFrame({"ITEM_ID": ["i1", "i2", "i3"]})
        result = Novelty(users_profile_df=profile, users_rec_list_df=rec, items_df=items).compute()
        self.assertGreaterEqual(result, 0.0)

    def test_compute_higher_novelty_for_rarer_items(self):
        """Recommending rarer items increases novelty."""
        profile_common = pd.DataFrame({
            "USER_ID": ["u1"] * 9 + ["u2"] * 1,
            "ITEM_ID": ["i1"] * 9 + ["i2"] * 1,
        })
        profile_rare = pd.DataFrame({
            "USER_ID": ["u1"] * 1 + ["u2"] * 1,
            "ITEM_ID": ["i1"] * 1 + ["i2"] * 1,
        })
        rec = pd.DataFrame({
            "USER_ID": ["u1", "u2"],
            "ITEM_ID": ["i1", "i1"],
            "ORDER": [1, 1],
        })
        items = pd.DataFrame({"ITEM_ID": ["i1", "i2"]})
        nov_common = Novelty(profile_common, rec, items).compute()
        nov_rare = Novelty(profile_rare, rec, items).compute()
        self.assertLessEqual(nov_common, nov_rare)


# ── Serendipity.single_process_serend ─────────────────────────────────────────

class TestSingleProcessSerend(unittest.TestCase):

    def _call(self, rec_ids, test_ids, baseline_ids):
        t3 = (0, pd.DataFrame({"ITEM_ID": baseline_ids}))
        t2 = (0, pd.DataFrame({"ITEM_ID": rec_ids}))
        t1 = (0, pd.DataFrame({"ITEM_ID": test_ids}))
        return Serendipity.single_process_serend(t3, t2, t1)

    def test_all_items_serendipitous_returns_one(self):
        """All recommended items are relevant and unexpected → 1.0."""
        result = self._call(rec_ids=[10], test_ids=[10], baseline_ids=[99])
        self.assertAlmostEqual(result, 1.0)

    def test_no_relevant_items_returns_zero(self):
        """No item is in the test set → serendipity = 0.0."""
        result = self._call(rec_ids=[10, 20], test_ids=[99, 98], baseline_ids=[50])
        self.assertAlmostEqual(result, 0.0)

    def test_all_expected_items_returns_zero(self):
        """All recommended items are in the baseline → 0.0 unexpectedness."""
        result = self._call(rec_ids=[10, 20], test_ids=[10, 20], baseline_ids=[10, 20])
        self.assertAlmostEqual(result, 0.0)

    def test_partial_serendipity(self):
        """One serendipitous item out of three → 1/3."""
        # rec=[10,20,30], test=[10,99], baseline=[20,30]
        # relevant = {10}; unexpected = {10}; serendipitous = {10}
        result = self._call(rec_ids=[10, 20, 30], test_ids=[10, 99], baseline_ids=[20, 30])
        self.assertAlmostEqual(result, 1 / 3)

    def test_result_in_zero_one_range(self):
        """Serendipity is always in [0, 1]."""
        result = self._call(rec_ids=[10, 20, 30], test_ids=[10, 30], baseline_ids=[10])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_unexpected_but_not_relevant_does_not_count(self):
        """Item not in baseline but also not in test set is not serendipitous."""
        # rec=[10,20], test=[99], baseline=[20]
        # useful = {}; unexpected = {10}; serendipitous = {}
        result = self._call(rec_ids=[10, 20], test_ids=[99], baseline_ids=[20])
        self.assertAlmostEqual(result, 0.0)

    def test_empty_rec_list_returns_zero(self):
        """Empty recommendation list → 0 (n_unexpected stays 0)."""
        result = self._call(rec_ids=[], test_ids=[10, 20], baseline_ids=[30])
        self.assertAlmostEqual(result, 0.0)

    def test_single_item_relevant_and_unexpected(self):
        """One item, relevant and unexpected → serendipity = 1.0."""
        result = self._call(rec_ids=[42], test_ids=[42], baseline_ids=[99])
        self.assertAlmostEqual(result, 1.0)


# ── Serendipity (integration) ──────────────────────────────────────────────────

class TestSerendipityCompute(unittest.TestCase):

    def test_compute_averaged_across_users(self):
        """Serendipity is the mean over all users."""
        rec = _rec_df((1, [10]), (2, [20]))
        test = _rec_df((1, [10]), (2, [99]))
        baseline = _rec_df((1, [99]), (2, [20]))
        # User 1: serendipitous → 1.0; User 2: 0.0
        result = Serendipity(
            users_rec_list_df=rec, users_test_df=test, users_baseline_df=baseline
        ).compute()
        self.assertAlmostEqual(result, 0.5)

    def test_compute_raises_on_user_mismatch(self):
        """Mismatched user sets in rec and test DataFrames raise IndexError."""
        rec = _rec_df((1, [10]), (2, [20]))
        test = _rec_df((1, [10]), (3, [30]))
        baseline = _rec_df((1, [99]), (2, [99]))
        with self.assertRaises(IndexError):
            Serendipity(rec, test, baseline).compute()


# ── Unexpectedness ─────────────────────────────────────────────────────────────

class TestUnexpectedness(unittest.TestCase):

    def _build(self, rec_pairs, test_pairs):
        return Unexpectedness(
            users_rec_list_df=_rec_df(*rec_pairs),
            users_test_df=_rec_df(*test_pairs),
        )

    def test_single_process_all_in_test_returns_zero(self):
        """All rec items are in the test set → unexpectedness = 0.0."""
        m = self._build([(1, [10, 20])], [(1, [10, 20, 30])])
        m.checking_users()
        m.ordering_and_grouping()
        groups2 = list(m.grouped_df_2)
        groups1 = list(m.grouped_df_1)
        result = m.single_process(groups2[0], groups1[0])
        self.assertAlmostEqual(result, 0.0)

    def test_single_process_none_in_test_returns_one(self):
        """No rec item in test set → unexpectedness = 1.0."""
        m = self._build([(1, [10, 20])], [(1, [99, 98])])
        m.checking_users()
        m.ordering_and_grouping()
        groups2 = list(m.grouped_df_2)
        groups1 = list(m.grouped_df_1)
        result = m.single_process(groups2[0], groups1[0])
        self.assertAlmostEqual(result, 1.0)

    def test_single_process_partial_overlap(self):
        """Half the items unexpected → unexpectedness = 0.5."""
        m = self._build([(1, [10, 20])], [(1, [10, 99])])
        m.checking_users()
        m.ordering_and_grouping()
        groups2 = list(m.grouped_df_2)
        groups1 = list(m.grouped_df_1)
        result = m.single_process(groups2[0], groups1[0])
        self.assertAlmostEqual(result, 0.5)

    def test_compute_mean_across_users(self):
        """Unexpectedness is averaged over all users."""
        m = self._build(
            [(1, [10, 20]), (2, [30, 40])],
            [(1, [10, 20]), (2, [99, 98])],
        )
        # User 1: 0.0; User 2: 1.0 → mean = 0.5
        self.assertAlmostEqual(m.compute(), 0.5)

    def test_compute_result_in_zero_one_range(self):
        """Unexpectedness must be in [0, 1]."""
        m = self._build(
            [(1, [10, 20, 30])],
            [(1, [10, 40])],
        )
        result = m.compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_compute_raises_on_user_mismatch(self):
        """Mismatched users raise IndexError."""
        rec = _rec_df((1, [10]))
        test = _rec_df((2, [10]))
        with self.assertRaises(IndexError):
            Unexpectedness(rec, test).compute()


# ── AverageNumberOfOItemsChanges ───────────────────────────────────────────────

class TestAverageNumberOfOItemsChanges(unittest.TestCase):

    def _build(self, rec_pairs, base_pairs):
        return AverageNumberOfOItemsChanges(
            users_rec_list_df=_rec_df(*rec_pairs, with_order=True),
            users_baseline_df=_rec_df(*base_pairs, with_order=True),
        )

    def test_single_process_no_changes_returns_zero(self):
        """Identical rec and baseline lists → 0 changes."""
        m = self._build([(1, [10, 20, 30])], [(1, [10, 20, 30])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        self.assertEqual(m.single_process(g2, g1), 0)

    def test_single_process_all_changed_returns_list_size(self):
        """Completely different rec vs baseline → number of new items = list size."""
        m = self._build([(1, [10, 20, 30])], [(1, [40, 50, 60])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        self.assertEqual(m.single_process(g2, g1), 3)

    def test_single_process_partial_change(self):
        """One item changed → single_process returns 1."""
        m = self._build([(1, [10, 20, 30])], [(1, [10, 20, 99])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        self.assertEqual(m.single_process(g2, g1), 1)

    def test_compute_averaged_across_users(self):
        """ANIC is averaged over all users."""
        m = self._build(
            [(1, [10, 20]), (2, [30, 40])],
            [(1, [10, 20]), (2, [99, 98])],
        )
        # User 1: 0 changes; User 2: 2 changes → mean = 1.0
        self.assertAlmostEqual(m.compute(), 1.0)

    def test_compute_result_non_negative(self):
        """ANIC is always ≥ 0."""
        m = self._build([(1, [10, 20, 30])], [(1, [10, 99, 88])])
        self.assertGreaterEqual(m.compute(), 0.0)

    def test_compute_raises_on_user_mismatch(self):
        """Mismatched users raise IndexError."""
        rec = _rec_df((1, [10]), with_order=True)
        base = _rec_df((2, [10]), with_order=True)
        with self.assertRaises(IndexError):
            AverageNumberOfOItemsChanges(rec, base).compute()


# ── AverageNumberOfGenreChanges ────────────────────────────────────────────────

class TestAverageNumberOfGenreChanges(unittest.TestCase):

    ITEMS = _items_df({10: "Action", 20: "Drama", 30: "Sci-Fi", 40: "Action|Drama"})

    def _build(self, rec_pairs, base_pairs):
        return AverageNumberOfGenreChanges(
            users_rec_list_df=_rec_df(*rec_pairs, with_order=True),
            users_baseline_df=_rec_df(*base_pairs, with_order=True),
            items_df=self.ITEMS,
        )

    def test_single_process_no_new_genres_returns_zero(self):
        """Rec introduces no genres not already in the baseline → 0."""
        m = self._build([(1, [10])], [(1, [10])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        self.assertEqual(m.single_process(g2, g1), 0)

    def test_single_process_entirely_new_genre(self):
        """Rec introduces a genre absent from the baseline → 1 new genre."""
        m = self._build([(1, [30])], [(1, [10])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        # baseline genres = {Action}; rec genres = {Sci-Fi}; new = 1
        self.assertEqual(m.single_process(g2, g1), 1)

    def test_single_process_multi_genre_item(self):
        """Item 40 (Action|Drama) shares Action with baseline item 10 → Drama is new."""
        m = self._build([(1, [40])], [(1, [10])])
        m.checking_users()
        m.ordering_and_grouping()
        g2, g1 = list(m.grouped_df_2)[0], list(m.grouped_df_1)[0]
        # baseline genres = {Action}; rec genres = {Action, Drama}; new = Drama → 1
        self.assertEqual(m.single_process(g2, g1), 1)

    def test_compute_averaged_across_users(self):
        """ANGC is the mean over all users."""
        m = self._build(
            [(1, [10]), (2, [30])],
            [(1, [10]), (2, [10])],
        )
        # User 1: 0 new genres; User 2: 1 new genre (Sci-Fi)
        self.assertAlmostEqual(m.compute(), 0.5)

    def test_compute_result_non_negative(self):
        """ANGC is always ≥ 0."""
        m = self._build([(1, [30, 20])], [(1, [10, 40])])
        self.assertGreaterEqual(m.compute(), 0.0)


# ── IntraListSimilarity ────────────────────────────────────────────────────────

# Pre-built encoded DataFrame: items A (Action only), B (Drama only), C (Action, same as A)
_ENCODED = pd.DataFrame(
    {"Action": [1, 0, 1], "Drama": [0, 1, 0]},
    index=["A", "B", "C"],
    dtype=float,
)


class TestIntraListSimilarity(unittest.TestCase):

    def _ils(self, rec_list_df=None):
        return IntraListSimilarity(
            users_rec_list_df=rec_list_df or pd.DataFrame(),
            items_df=pd.DataFrame(),
            encoded_df=_ENCODED,
        )

    def test_single_list_similarity_identical_genre_vectors_returns_one(self):
        """Two items with the same genre vector → cosine similarity = 1.0."""
        ils = self._ils()
        result = ils._single_list_similarity(["A", "C"])
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_single_list_similarity_orthogonal_genre_vectors_returns_zero(self):
        """Two items with orthogonal genre vectors → cosine similarity = 0.0."""
        ils = self._ils()
        result = ils._single_list_similarity(["A", "B"])
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_single_list_similarity_three_items_mixed(self):
        """A, B, C → pairs: A-B=0, A-C=1, B-C=0 → mean = 1/3."""
        ils = self._ils()
        result = ils._single_list_similarity(["A", "B", "C"])
        self.assertAlmostEqual(result, 1 / 3, places=5)

    def test_single_list_similarity_result_in_zero_one_range(self):
        """ILS for any list must be in [0, 1]."""
        ils = self._ils()
        for lst in [["A", "B"], ["A", "C"], ["A", "B", "C"]]:
            result = ils._single_list_similarity(lst)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_compute_mean_across_users(self):
        """ILS is averaged over all users."""
        # User 1: [A, C] → ILS = 1.0; User 2: [A, B] → ILS = 0.0 → mean = 0.5
        rec = pd.DataFrame({
            "USER_ID": [1, 1, 2, 2],
            "ITEM_ID": ["A", "C", "A", "B"],
        })
        result = IntraListSimilarity(
            users_rec_list_df=rec, items_df=pd.DataFrame(), encoded_df=_ENCODED
        ).compute()
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_compute_all_same_genres_returns_one(self):
        """When every user's list contains only identical-genre items, ILS = 1.0."""
        rec = pd.DataFrame({
            "USER_ID": [1, 1],
            "ITEM_ID": ["A", "C"],
        })
        result = IntraListSimilarity(
            users_rec_list_df=rec, items_df=pd.DataFrame(), encoded_df=_ENCODED
        ).compute()
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_compute_all_orthogonal_items_returns_zero(self):
        """When every user's list has mutually orthogonal items, ILS = 0.0."""
        rec = pd.DataFrame({
            "USER_ID": [1, 1],
            "ITEM_ID": ["A", "B"],
        })
        result = IntraListSimilarity(
            users_rec_list_df=rec, items_df=pd.DataFrame(), encoded_df=_ENCODED
        ).compute()
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_compute_result_in_zero_one_range(self):
        """ILS must be in [0, 1] for any valid input."""
        rec = pd.DataFrame({
            "USER_ID": [1, 1, 1],
            "ITEM_ID": ["A", "B", "C"],
        })
        result = IntraListSimilarity(
            users_rec_list_df=rec, items_df=pd.DataFrame(), encoded_df=_ENCODED
        ).compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
