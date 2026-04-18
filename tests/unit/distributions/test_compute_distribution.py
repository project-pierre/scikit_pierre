"""
Unit tests for scikit_pierre.distributions.compute_distribution.
"""
import unittest

import pandas as pd
import pytest

from scikit_pierre.distributions.compute_distribution import (
    computer_users_distribution_dict,
    computer_users_distribution_pandas,
    transform_to_vec,
)


# ── transform_to_vec ───────────────────────────────────────────────────────────

class TestTransformToVec(unittest.TestCase):

    def test_identical_keys_no_zeros_inserted(self):
        """When both dicts share the same keys, no padding is needed."""
        p_dict = {"Action": 0.6, "Drama": 0.4}
        q_dict = {"Action": 0.3, "Drama": 0.7}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertEqual(len(p), 2)
        self.assertEqual(len(q), 2)

    def test_output_length_equals_union_of_keys(self):
        """Result length equals the number of unique genre keys across both dicts."""
        p_dict = {"Action": 0.5, "Drama": 0.5}
        q_dict = {"Comedy": 0.5, "Drama": 0.5}
        p, q = transform_to_vec(p_dict, q_dict)
        # union = {Action, Drama, Comedy} → length 3
        self.assertEqual(len(p), 3)
        self.assertEqual(len(q), 3)

    def test_missing_key_in_p_gets_zero(self):
        """A genre in q but not in p yields 0.0 in p's vector."""
        p_dict = {"Action": 1.0}
        q_dict = {"Drama": 1.0}
        p, q = transform_to_vec(p_dict, q_dict)
        # p should contain [1.0, 0.0] and q [0.0, 1.0] in some order
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertAlmostEqual(sum(q), 1.0)
        self.assertIn(0.0, p)
        self.assertIn(0.0, q)

    def test_missing_key_in_q_gets_zero(self):
        """A genre in p but not in q yields 0.0 in q's vector."""
        p_dict = {"Action": 0.5, "Drama": 0.5}
        q_dict = {"Action": 1.0}
        p, q = transform_to_vec(p_dict, q_dict)
        # Drama is missing in q → one entry in q is 0.0
        self.assertIn(0.0, q)

    def test_disjoint_supports_both_vectors_have_zeros(self):
        """Completely disjoint dicts produce vectors with zeros on both sides."""
        p_dict = {"Action": 1.0}
        q_dict = {"Drama": 1.0}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertIn(0.0, p)
        self.assertIn(0.0, q)

    def test_p_and_q_same_length(self):
        """The two returned lists always have the same length."""
        p_dict = {"A": 0.3, "B": 0.3, "C": 0.4}
        q_dict = {"B": 0.5, "D": 0.5}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertEqual(len(p), len(q))

    def test_empty_target_dist(self):
        """An empty target dict yields all-zero p."""
        p_dict = {}
        q_dict = {"Action": 0.5, "Drama": 0.5}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertTrue(all(v == 0.0 for v in p))
        self.assertEqual(len(q), 2)

    def test_empty_realized_dist(self):
        """An empty realized dict yields all-zero q."""
        p_dict = {"Action": 1.0}
        q_dict = {}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertTrue(all(v == 0.0 for v in q))

    def test_values_are_floats(self):
        """All returned values are floats."""
        p_dict = {"Action": 1}
        q_dict = {"Action": 1}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertIsInstance(p[0], float)
        self.assertIsInstance(q[0], float)

    def test_golden_single_shared_key(self):
        """Single shared key: p=[0.7], q=[0.3] — values are preserved."""
        p_dict = {"Action": 0.7}
        q_dict = {"Action": 0.3}
        p, q = transform_to_vec(p_dict, q_dict)
        self.assertEqual(len(p), 1)
        self.assertAlmostEqual(p[0], 0.7)
        self.assertAlmostEqual(q[0], 0.3)

    def test_determinism_same_output_on_repeated_calls(self):
        """Identical inputs always produce identical results."""
        p_dict = {"Action": 0.4, "Drama": 0.6}
        q_dict = {"Drama": 0.5, "Comedy": 0.5}
        p1, q1 = transform_to_vec(p_dict, q_dict)
        p2, q2 = transform_to_vec(p_dict, q_dict)
        self.assertEqual(p1, p2)
        self.assertEqual(q1, q2)


# ── computer_users_distribution_dict ──────────────────────────────────────────

def _make_interactions(rows):
    """Build USER_ID / ITEM_ID / TRANSACTION_VALUE DataFrame."""
    return pd.DataFrame(rows, columns=["USER_ID", "ITEM_ID", "TRANSACTION_VALUE"])


def _make_items(rows):
    """Build ITEM_ID / GENRES DataFrame."""
    return pd.DataFrame(rows, columns=["ITEM_ID", "GENRES"])


class TestComputerUserDistributionDict(unittest.TestCase):

    def setUp(self):
        self.interactions = _make_interactions([
            (1, "i1", 5.0),
            (1, "i2", 3.0),
            (2, "i1", 4.0),
        ])
        self.items = _make_items([
            ("i1", "Action"),
            ("i2", "Drama"),
        ])

    def test_returns_dict(self):
        """Return type is a dict."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        self.assertIsInstance(result, dict)

    def test_keys_are_user_ids(self):
        """Dict keys correspond to the unique USER_IDs in the interactions."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_values_are_dicts(self):
        """Each per-user value is itself a dict mapping genre → float."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        for uid, dist in result.items():
            self.assertIsInstance(dist, dict)

    def test_genre_keys_are_strings(self):
        """Genre keys in each per-user distribution are strings."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        for dist in result.values():
            for k in dist.keys():
                self.assertIsInstance(k, str)

    def test_distribution_values_non_negative(self):
        """All distribution values are ≥ 0."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        for dist in result.values():
            for v in dist.values():
                self.assertGreaterEqual(v, 0.0)

    def test_wps_normalizes_to_one(self):
        """WPS (probability-normalised) distributions sum to 1.0 per user."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="WPS"
        )
        for uid, dist in result.items():
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=9,
                                   msg=f"User {uid} WPS does not sum to 1")

    def test_pgd_p_normalizes_to_one(self):
        """PGD_P distributions sum to 1.0 per user."""
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="PGD_P"
        )
        for dist in result.values():
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_single_genre_item_cws_returns_positive(self):
        """Single-interaction user gets a positive CWS value for their genre."""
        interactions = _make_interactions([(1, "i1", 5.0)])
        items = _make_items([("i1", "Action")])
        result = computer_users_distribution_dict(
            interactions, items, distribution="CWS"
        )
        self.assertIn("Action", result[1])
        self.assertGreater(result[1]["Action"], 0.0)

    def test_two_users_produce_independent_distributions(self):
        """Each user's distribution reflects only their own interactions."""
        # User 2 has only item i1 (Action); should not have Drama
        result = computer_users_distribution_dict(
            self.interactions, self.items, distribution="CWS"
        )
        # user 2 interacted only with i1 (Action) → Drama should not appear
        self.assertNotIn("Drama", result[2])

    def test_unknown_distribution_raises_name_error(self):
        """Passing an unknown distribution acronym raises NameError."""
        with self.assertRaises(NameError):
            computer_users_distribution_dict(
                self.interactions, self.items, distribution="BOGUS"
            )


# ── computer_users_distribution_pandas ────────────────────────────────────────

class TestComputerUserDistributionPandas(unittest.TestCase):

    def setUp(self):
        self.interactions = _make_interactions([
            ("u1", "i1", 5.0),
            ("u1", "i2", 3.0),
            ("u2", "i1", 4.0),
        ])
        self.items = _make_items([
            ("i1", "Action"),
            ("i2", "Drama"),
        ])

    def test_returns_dataframe(self):
        """Return type is a DataFrame."""
        import pandas as pd
        result = computer_users_distribution_pandas(
            self.interactions, self.items, distribution="CWS"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_row_count_equals_user_count(self):
        """One row per unique user in the interactions."""
        result = computer_users_distribution_pandas(
            self.interactions, self.items, distribution="CWS"
        )
        self.assertEqual(len(result), 2)

    def test_column_names_are_genres(self):
        """Columns correspond to genre labels."""
        result = computer_users_distribution_pandas(
            self.interactions, self.items, distribution="CWS"
        )
        self.assertIn("Action", result.columns)

    def test_wps_rows_sum_to_one(self):
        """WPS rows (proper probability distributions) sum to 1.0."""
        result = computer_users_distribution_pandas(
            self.interactions, self.items, distribution="WPS"
        )
        for _, row in result.iterrows():
            self.assertAlmostEqual(row.sum(), 1.0, places=9)
