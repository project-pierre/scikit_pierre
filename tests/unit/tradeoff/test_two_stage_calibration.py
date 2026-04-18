"""
Unit tests for TwoStageCalibration (popularity → genre pipeline).

Covers construction validation, config(), and fit() output contracts.
"""
import pytest
import pandas as pd

from scikit_pierre.tradeoff.calibration import TwoStageCalibration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs():
    """3 users × 4 items each → avg_profile = 4, stage1 list_size = ceil(2) = 2."""
    rows = []
    for uid in [1, 2, 3]:
        for iid in [10, 20, 30, 40]:
            rows.append({"USER_ID": uid, "ITEM_ID": iid, "TRANSACTION_VALUE": float(iid / 10)})
    return pd.DataFrame(rows)


@pytest.fixture
def cands():
    """Each user has 5 candidates."""
    rows = []
    for uid in [1, 2, 3]:
        for iid, score in zip([10, 20, 30, 40, 50], [5.0, 4.0, 3.0, 2.0, 1.0]):
            rows.append({"USER_ID": uid, "ITEM_ID": iid, "TRANSACTION_VALUE": score})
    return pd.DataFrame(rows)


@pytest.fixture
def items():
    """5-item catalogue with GENRES column."""
    return pd.DataFrame({
        "ITEM_ID": [10, 20, 30, 40, 50],
        "GENRES": ["Action|Comedy", "Drama", "Action", "Comedy", "Thriller"],
    })


@pytest.fixture
def tsc(prefs, cands, items):
    inst = TwoStageCalibration(prefs, cands, items)
    inst.config(
        distribution_component="CWS",
        fairness_component="KL",
        relevance_component="SUM",
        tradeoff_weight_component="VAR",
        select_item_component="SURROGATE",
        list_size=3,
    )
    return inst


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTwoStageCalibrationConstruction:

    def test_valid_construction_succeeds(self, prefs, cands, items):
        inst = TwoStageCalibration(prefs, cands, items)
        assert inst is not None

    def test_missing_genres_column_raises(self, prefs, cands):
        """item_set without GENRES or POPULARITY must raise."""
        bad_items = pd.DataFrame({"ITEM_ID": [10, 20], "TITLE": ["A", "B"]})
        with pytest.raises(Exception):
            TwoStageCalibration(prefs, cands, bad_items)

    def test_config_stores_list_size(self, prefs, cands, items):
        inst = TwoStageCalibration(prefs, cands, items)
        inst.config(list_size=5)
        assert inst._list_size == 5


# ---------------------------------------------------------------------------
# fit() output contract
# ---------------------------------------------------------------------------

class TestTwoStageCalibrationFit:

    def test_fit_returns_dataframe(self, tsc):
        result = tsc.fit()
        assert isinstance(result, pd.DataFrame)

    def test_fit_has_user_id_column(self, tsc):
        result = tsc.fit()
        assert "USER_ID" in result.columns

    def test_fit_has_item_id_column(self, tsc):
        result = tsc.fit()
        assert "ITEM_ID" in result.columns

    def test_fit_list_size_per_user(self, tsc):
        """Each user must have at most list_size rows in the final output."""
        result = tsc.fit()
        for uid, group in result.groupby("USER_ID"):
            assert len(group) <= 3

    def test_fit_all_users_in_output(self, tsc, prefs):
        result = tsc.fit()
        assert set(prefs["USER_ID"].unique()).issubset(set(result["USER_ID"].unique()))

    def test_fit_no_duplicate_items_per_user(self, tsc):
        result = tsc.fit()
        for uid, group in result.groupby("USER_ID"):
            assert group["ITEM_ID"].nunique() == len(group)

    def test_fit_without_config_raises(self, prefs, cands, items):
        """fit() before config() must raise (no environment set on sub-instances)."""
        inst = TwoStageCalibration(prefs, cands, items)
        with pytest.raises(Exception):
            inst.fit()
