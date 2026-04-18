"""
Unit tests for scikit_pierre.metrics.base.
"""
import unittest

import pandas as pd

from scikit_pierre.metrics.base import BaseMetric


def _make_df(*user_item_pairs):
    """Build a USER_ID/ITEM_ID DataFrame from (user, item) pairs."""
    rows = [{"USER_ID": u, "ITEM_ID": i} for u, i in user_item_pairs]
    return pd.DataFrame(rows)


def _group_tuple(df, user_id):
    """Return the (user_id, sub_df) tuple that groupby would produce."""
    return user_id, df[df["USER_ID"] == user_id].reset_index(drop=True)


# ── get_bool_list ──────────────────────────────────────────────────────────────

class TestGetBoolList(unittest.TestCase):

    def test_get_bool_list_full_overlap_returns_all_true(self):
        """Every rec item is in the test set."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 20, 30]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10, 20, 30]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [True, True, True])

    def test_get_bool_list_no_overlap_returns_all_false(self):
        """No rec item is in the test set."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 20, 30]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [40, 50, 60]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [False, False, False])

    def test_get_bool_list_partial_overlap(self):
        """Only some rec items appear in the test set."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 20, 30]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10, 30, 40]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [True, False, True])

    def test_get_bool_list_empty_rec_returns_empty(self):
        """Empty recommendation list yields an empty bool list."""
        rec = (1, pd.DataFrame({"ITEM_ID": pd.Series([], dtype=int)}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10, 20]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [])

    def test_get_bool_list_empty_test_returns_all_false(self):
        """Empty test set means no rec item can be relevant."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 20]}))
        tst = (1, pd.DataFrame({"ITEM_ID": pd.Series([], dtype=int)}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [False, False])

    def test_get_bool_list_single_match(self):
        """Single item that is relevant."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [True])

    def test_get_bool_list_single_no_match(self):
        """Single item that is not relevant."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [99]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [False])

    def test_get_bool_list_preserves_rec_order(self):
        """Result order follows the recommendation list, not the test set."""
        rec = (1, pd.DataFrame({"ITEM_ID": [30, 10, 20]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10, 20]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [False, True, True])

    def test_get_bool_list_duplicate_rec_items_each_checked_independently(self):
        """Duplicate item IDs in the rec list are each evaluated separately."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 10, 20]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10]}))
        self.assertEqual(BaseMetric.get_bool_list(rec, tst), [True, True, False])

    def test_get_bool_list_length_equals_rec_list_length(self):
        """The result always has the same length as the rec list."""
        rec = (1, pd.DataFrame({"ITEM_ID": [10, 20, 30, 40, 50]}))
        tst = (1, pd.DataFrame({"ITEM_ID": [10, 30]}))
        result = BaseMetric.get_bool_list(rec, tst)
        self.assertEqual(len(result), 5)


# ── checking_users ─────────────────────────────────────────────────────────────

class TestCheckingUsers(unittest.TestCase):

    def test_checking_users_matching_users_does_not_raise(self):
        """Identical user sets in df_1 and df_2 passes silently."""
        df1 = pd.DataFrame({"USER_ID": [1, 2]})
        df2 = pd.DataFrame({"USER_ID": [2, 1]})
        BaseMetric(df_1=df1, df_2=df2).checking_users()

    def test_checking_users_mismatched_set_raises_index_error(self):
        """Different user sets raise IndexError."""
        df1 = pd.DataFrame({"USER_ID": [1, 2]})
        df2 = pd.DataFrame({"USER_ID": [1, 3]})
        with self.assertRaises(IndexError):
            BaseMetric(df_1=df1, df_2=df2).checking_users()

    def test_checking_users_subset_raises_index_error(self):
        """df_2 being a strict subset of df_1's users raises IndexError."""
        df1 = pd.DataFrame({"USER_ID": [1, 2, 3]})
        df2 = pd.DataFrame({"USER_ID": [1, 2]})
        with self.assertRaises(IndexError):
            BaseMetric(df_1=df1, df_2=df2).checking_users()

    def test_checking_users_superset_raises_index_error(self):
        """df_2 containing extra users compared to df_1 raises IndexError."""
        df1 = pd.DataFrame({"USER_ID": [1, 2]})
        df2 = pd.DataFrame({"USER_ID": [1, 2, 3]})
        with self.assertRaises(IndexError):
            BaseMetric(df_1=df1, df_2=df2).checking_users()

    def test_checking_users_single_user_matching(self):
        """Single matching user passes silently."""
        df1 = pd.DataFrame({"USER_ID": [42]})
        df2 = pd.DataFrame({"USER_ID": [42]})
        BaseMetric(df_1=df1, df_2=df2).checking_users()

    def test_checking_users_compares_as_strings(self):
        """User IDs are cast to str before comparison, so int 1 == str '1'."""
        df1 = pd.DataFrame({"USER_ID": [1]})
        df2 = pd.DataFrame({"USER_ID": ["1"]})
        # Should not raise because both cast to "1"
        BaseMetric(df_1=df1, df_2=df2).checking_users()


# ── ordering ───────────────────────────────────────────────────────────────────

class TestOrdering(unittest.TestCase):

    def test_ordering_sorts_df1_by_user_id(self):
        """df_1 is sorted in ascending USER_ID order after ordering()."""
        df1 = pd.DataFrame({"USER_ID": [3, 1, 2], "ITEM_ID": [30, 10, 20]})
        m = BaseMetric(df_1=df1)
        m.ordering()
        self.assertEqual(m.df_1["USER_ID"].tolist(), [1, 2, 3])

    def test_ordering_sorts_df2_by_user_id(self):
        """df_2 is sorted in ascending USER_ID order after ordering()."""
        df2 = pd.DataFrame({"USER_ID": [3, 1, 2], "ITEM_ID": [30, 10, 20]})
        m = BaseMetric(df_2=df2)
        m.ordering()
        self.assertEqual(m.df_2["USER_ID"].tolist(), [1, 2, 3])

    def test_ordering_sorts_df3_by_user_id(self):
        """df_3 is sorted in ascending USER_ID order after ordering()."""
        df3 = pd.DataFrame({"USER_ID": [2, 1], "ITEM_ID": [20, 10]})
        m = BaseMetric(df_3=df3)
        m.ordering()
        self.assertEqual(m.df_3["USER_ID"].tolist(), [1, 2])

    def test_ordering_skips_none_dataframes_without_error(self):
        """ordering() is a no-op for None DataFrames."""
        m = BaseMetric(df_1=None, df_2=None, df_3=None)
        m.ordering()  # must not raise


# ── grouping ───────────────────────────────────────────────────────────────────

class TestGrouping(unittest.TestCase):

    def test_grouping_creates_correct_number_of_groups(self):
        """grouped_df_1 contains exactly as many groups as unique users."""
        df1 = pd.DataFrame({"USER_ID": [1, 1, 2, 2], "ITEM_ID": [10, 20, 30, 40]})
        m = BaseMetric(df_1=df1)
        m.grouping()
        self.assertEqual(len(m.grouped_df_1), 2)

    def test_grouping_skips_none_df_leaves_grouped_as_none(self):
        """grouped_df_1 stays None when df_1 is None."""
        m = BaseMetric(df_1=None)
        m.grouping()
        self.assertIsNone(m.grouped_df_1)

    def test_grouping_groups_all_three_dfs(self):
        """All three grouped attributes are populated when all dfs are provided."""
        df = pd.DataFrame({"USER_ID": [1, 2], "ITEM_ID": [10, 20]})
        m = BaseMetric(df_1=df.copy(), df_2=df.copy(), df_3=df.copy())
        m.grouping()
        self.assertIsNotNone(m.grouped_df_1)
        self.assertIsNotNone(m.grouped_df_2)
        self.assertIsNotNone(m.grouped_df_3)


# ── ordering_and_grouping ──────────────────────────────────────────────────────

class TestOrderingAndGrouping(unittest.TestCase):

    def test_ordering_and_grouping_produces_sorted_groups(self):
        """After ordering_and_grouping, groups are in ascending USER_ID order."""
        df1 = pd.DataFrame({"USER_ID": [2, 1], "ITEM_ID": [20, 10]})
        df2 = pd.DataFrame({"USER_ID": [2, 1], "ITEM_ID": [20, 10]})
        m = BaseMetric(df_1=df1, df_2=df2)
        m.ordering_and_grouping()
        group_keys_1 = [k for k, _ in m.grouped_df_1]
        group_keys_2 = [k for k, _ in m.grouped_df_2]
        self.assertEqual(group_keys_1, sorted(group_keys_1))
        self.assertEqual(group_keys_2, sorted(group_keys_2))
