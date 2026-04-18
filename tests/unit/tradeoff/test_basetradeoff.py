"""
Unit tests for scikit_pierre.tradeoff.basetradeoff.BaseTradeOff.

Covers constructor validation (column checks, item-set membership) and
the env/fit guard contract.
"""
import pytest
import pandas as pd

from scikit_pierre.tradeoff.basetradeoff import BaseTradeOff


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def users_prefs():
    return pd.DataFrame({
        "USER_ID": [1, 1, 2, 2],
        "ITEM_ID": [10, 20, 10, 30],
        "TRANSACTION_VALUE": [4.0, 3.0, 5.0, 2.0],
    })


@pytest.fixture
def candidate_items():
    return pd.DataFrame({
        "USER_ID": [1, 1, 2, 2],
        "ITEM_ID": [20, 30, 20, 30],
        "TRANSACTION_VALUE": [3.5, 2.5, 4.5, 1.5],
    })


@pytest.fixture
def item_set():
    return pd.DataFrame({
        "ITEM_ID": [10, 20, 30],
        "GENRES": ["Action|Comedy", "Drama", "Action"],
    })


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

class TestBaseTradeOffInit:

    def test_valid_construction_stores_dataframes(self, users_prefs, candidate_items, item_set):
        """Constructor must store copies of valid DataFrames without raising."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert list(obj.users_preferences.columns) == list(users_prefs.columns)
        assert list(obj.candidate_items.columns) == list(candidate_items.columns)

    def test_construction_stores_item_set(self, users_prefs, candidate_items, item_set):
        """Constructor must store a copy of item_set."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert obj.item_set is not item_set

    def test_environment_starts_empty(self, users_prefs, candidate_items, item_set):
        """environment dict must be empty before env() is called."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert obj.environment == {}

    def test_users_distribution_none_by_default(self, users_prefs, candidate_items, item_set):
        """users_distribution is None when not supplied."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert obj.users_distribution is None

    def test_batch_defaults_to_128(self, users_prefs, candidate_items, item_set):
        """Default batch size is 128."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert obj.batch == 128

    def test_custom_batch_stored(self, users_prefs, candidate_items, item_set):
        """Custom batch value is stored correctly."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set, batch=32)
        assert obj.batch == 32

    def test_input_dataframes_are_deep_copied(self, users_prefs, candidate_items, item_set):
        """Mutations to the original DataFrames must not affect stored copies."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        users_prefs.iloc[0, 2] = 999.0
        assert obj.users_preferences.iloc[0, 2] != 999.0


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

class TestBaseTradeOffColumnValidation:

    def test_missing_user_id_raises_key_error(self, item_set):
        """Both DataFrames missing USER_ID must raise KeyError."""
        bad = pd.DataFrame({"ITEM_ID": [10], "TRANSACTION_VALUE": [1.0]})
        with pytest.raises(KeyError):
            BaseTradeOff(bad, bad, item_set)

    def test_missing_item_id_raises_key_error(self, candidate_items, item_set):
        """users_preferences missing ITEM_ID must raise KeyError."""
        bad = pd.DataFrame({
            "USER_ID": [1], "TRANSACTION_VALUE": [1.0]
        })
        with pytest.raises(KeyError):
            BaseTradeOff(bad, candidate_items, item_set)

    def test_missing_transaction_value_raises_key_error(self, candidate_items, item_set):
        """Both DataFrames missing TRANSACTION_VALUE must raise KeyError."""
        bad_prefs = pd.DataFrame({"USER_ID": [1], "ITEM_ID": [10]})
        bad_cands = pd.DataFrame({"USER_ID": [1], "ITEM_ID": [10]})
        with pytest.raises(KeyError):
            BaseTradeOff(bad_prefs, bad_cands, item_set)

    def test_candidate_items_with_transaction_value_is_sufficient(self, item_set):
        """If candidate_items has the required columns, construction succeeds
        even when users_preferences lacks TRANSACTION_VALUE."""
        prefs_no_tv = pd.DataFrame({"USER_ID": [1], "ITEM_ID": [10]})
        cands_ok = pd.DataFrame({
            "USER_ID": [1], "ITEM_ID": [10], "TRANSACTION_VALUE": [1.0]
        })
        item_set_mini = pd.DataFrame({"ITEM_ID": [10], "GENRES": ["Action"]})
        obj = BaseTradeOff(prefs_no_tv, cands_ok, item_set_mini)
        assert obj is not None


# ---------------------------------------------------------------------------
# Item-set membership validation
# ---------------------------------------------------------------------------

class TestBaseTradeOffItemSetValidation:

    def test_missing_item_in_item_set_raises_name_error(self, users_prefs, item_set):
        """Candidate items referencing IDs absent from item_set must raise NameError."""
        cands_with_unknown = pd.DataFrame({
            "USER_ID": [1], "ITEM_ID": [999], "TRANSACTION_VALUE": [1.0]
        })
        with pytest.raises(NameError):
            BaseTradeOff(users_prefs, cands_with_unknown, item_set)

    def test_all_ids_present_no_error(self, users_prefs, candidate_items, item_set):
        """When all referenced IDs exist in item_set, no exception is raised."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        assert obj is not None


# ---------------------------------------------------------------------------
# env() and fit() guard
# ---------------------------------------------------------------------------

class TestBaseTradeOffEnvAndFit:

    def test_env_stores_configuration(self, users_prefs, candidate_items, item_set):
        """env() must store the supplied dict into self.environment."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        cfg = {"list_size": 5, "distribution": "CWS"}
        obj.env(cfg)
        assert obj.environment == cfg

    def test_fit_without_env_raises_system_error(self, users_prefs, candidate_items, item_set):
        """fit() must raise SystemError when environment is empty."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        with pytest.raises(SystemError):
            obj.fit()

    def test_fit_after_env_does_not_raise(self, users_prefs, candidate_items, item_set):
        """fit() must not raise once env() has populated self.environment."""
        obj = BaseTradeOff(users_prefs, candidate_items, item_set)
        obj.env({"list_size": 5})
        obj.fit()  # should complete without exception
