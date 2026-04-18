"""
Unit tests for LogarithmBias calibration class.

Covers construction, bias computation, config/fit pipeline,
and output schema contracts.
"""
import pytest
import pandas as pd

from scikit_pierre.tradeoff.calibration import LogarithmBias


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs():
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 2, 2, 2],
        "ITEM_ID": [10, 20, 30, 10, 20, 40],
        "TRANSACTION_VALUE": [5.0, 4.0, 3.0, 4.0, 3.0, 5.0],
    })


@pytest.fixture
def cands():
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 1, 2, 2, 2, 2],
        "ITEM_ID": [10, 20, 30, 40, 10, 20, 30, 40],
        "TRANSACTION_VALUE": [5.0, 4.0, 3.0, 2.0, 4.0, 3.0, 2.5, 5.0],
    })


@pytest.fixture
def items():
    return pd.DataFrame({
        "ITEM_ID": [10, 20, 30, 40],
        "GENRES": ["Action|Comedy", "Drama", "Action", "Comedy"],
    })


@pytest.fixture
def lb(prefs, cands, items):
    inst = LogarithmBias(prefs, cands, items)
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

class TestLogarithmBiasConstruction:

    def test_valid_construction_succeeds(self, prefs, cands, items):
        """Constructor must not raise with valid DataFrames."""
        inst = LogarithmBias(prefs, cands, items)
        assert inst is not None

    def test_item_bias_none_before_fit(self, prefs, cands, items):
        """item_bias is None until fit() is called."""
        inst = LogarithmBias(prefs, cands, items)
        assert inst.item_bias is None


# ---------------------------------------------------------------------------
# _computing_item_bias
# ---------------------------------------------------------------------------

class TestComputingItemBias:

    def test_bias_returns_dataframe(self, lb, prefs):
        """_computing_item_bias must return a DataFrame."""
        lb.transaction_mean = prefs["TRANSACTION_VALUE"].mean()
        bias_df = lb._computing_item_bias(prefs)
        assert isinstance(bias_df, pd.DataFrame)

    def test_bias_has_correct_columns(self, lb, prefs):
        """Bias DataFrame must have ITEM_ID and BIAS_VALUE columns."""
        lb.transaction_mean = prefs["TRANSACTION_VALUE"].mean()
        bias_df = lb._computing_item_bias(prefs)
        assert "ITEM_ID" in bias_df.columns
        assert "BIAS_VALUE" in bias_df.columns

    def test_bias_covers_all_items(self, lb, prefs):
        """One bias row per unique item in users_preferences."""
        lb.transaction_mean = prefs["TRANSACTION_VALUE"].mean()
        bias_df = lb._computing_item_bias(prefs)
        assert len(bias_df) == prefs["ITEM_ID"].nunique()

    def test_bias_values_are_finite(self, lb, prefs):
        """All BIAS_VALUE entries must be finite floats."""
        lb.transaction_mean = prefs["TRANSACTION_VALUE"].mean()
        bias_df = lb._computing_item_bias(prefs)
        assert bias_df["BIAS_VALUE"].apply(lambda v: isinstance(v, float) and
                                           not (v != v)).all()


# ---------------------------------------------------------------------------
# fit() output contract
# ---------------------------------------------------------------------------

class TestLogarithmBiasFit:

    def test_fit_returns_dataframe(self, lb):
        result = lb.fit()
        assert isinstance(result, pd.DataFrame)

    def test_fit_output_has_user_id(self, lb):
        result = lb.fit()
        assert "USER_ID" in result.columns

    def test_fit_output_has_item_id(self, lb):
        result = lb.fit()
        assert "ITEM_ID" in result.columns

    def test_fit_list_size_per_user(self, lb):
        """Each user must have at most list_size rows."""
        result = lb.fit()
        for uid, group in result.groupby("USER_ID"):
            assert len(group) <= 3

    def test_fit_no_duplicate_items_per_user(self, lb):
        result = lb.fit()
        for uid, group in result.groupby("USER_ID"):
            assert group["ITEM_ID"].nunique() == len(group)

    def test_fit_without_config_raises(self, prefs, cands, items):
        """fit() before config() must raise SystemError."""
        inst = LogarithmBias(prefs, cands, items)
        with pytest.raises(SystemError):
            inst.fit()
