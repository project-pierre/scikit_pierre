"""
Unit and integration tests for LinearCalibration.

Covers construction, config(), fit() output contracts, edge cases,
invalid inputs, idempotent reconfiguration, and mathematical properties.
"""
import pytest
import random
import numpy as np
import pandas as pd
import pandas.testing as tm

from scikit_pierre.tradeoff.calibration import LinearCalibration


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_prefs():
    """2 users, 3 items each, deterministic scores."""
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 2, 2, 2],
        "ITEM_ID": [10, 20, 30, 10, 20, 40],
        "TRANSACTION_VALUE": [5.0, 4.0, 3.0, 4.0, 3.0, 5.0],
    })


@pytest.fixture
def small_candidates():
    """Each user has 5 candidates (all known items)."""
    return pd.DataFrame({
        "USER_ID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "ITEM_ID": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
        "TRANSACTION_VALUE": [5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.5, 5.0, 1.5],
    })


@pytest.fixture
def small_items():
    """5-item catalogue with multi-genre and single-genre entries."""
    return pd.DataFrame({
        "ITEM_ID": [10, 20, 30, 40, 50],
        "GENRES": ["Action|Comedy", "Drama", "Action", "Comedy|Drama", "Thriller"],
    })


@pytest.fixture
def lc(small_prefs, small_candidates, small_items):
    """Configured LinearCalibration instance ready to call fit()."""
    inst = LinearCalibration(small_prefs, small_candidates, small_items)
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
# Construction tests
# ---------------------------------------------------------------------------

class TestLinearCalibrationConstruction:

    def test_valid_construction_succeeds(self, small_prefs, small_candidates, small_items):
        """Constructor must not raise with valid DataFrames."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        assert inst is not None

    def test_components_are_none_before_config(self, small_prefs, small_candidates, small_items):
        """All component attributes are None until config() is called."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        assert inst._distribution_component is None
        assert inst._fairness_component is None
        assert inst._relevance_component is None
        assert inst._tradeoff_weight_component is None
        assert inst._select_item_component is None

    def test_non_dataframe_users_prefs_raises(self, small_candidates, small_items):
        """Passing a dict instead of DataFrame must raise."""
        with pytest.raises(Exception):
            LinearCalibration({"USER_ID": [1]}, small_candidates, small_items)

    def test_non_dataframe_candidate_items_raises(self, small_prefs, small_items):
        """Passing a list instead of DataFrame for candidate_items must raise."""
        with pytest.raises(Exception):
            LinearCalibration(small_prefs, [[1, 2, 3]], small_items)

    def test_item_id_missing_from_item_set_raises(self, small_prefs, small_items):
        """candidate_items referencing unknown item IDs must raise NameError."""
        bad_cands = pd.DataFrame({
            "USER_ID": [1], "ITEM_ID": [999], "TRANSACTION_VALUE": [1.0]
        })
        with pytest.raises(NameError):
            LinearCalibration(small_prefs, bad_cands, small_items)


# ---------------------------------------------------------------------------
# config() tests
# ---------------------------------------------------------------------------

class TestLinearCalibrationConfig:

    def test_config_resolves_distribution_component(
            self, small_prefs, small_candidates, small_items):
        """config() must resolve distribution_component to a callable."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(distribution_component="CWS")
        assert callable(inst._distribution_component)

    def test_config_resolves_fairness_component(
            self, small_prefs, small_candidates, small_items):
        """config() must resolve fairness_component to a callable."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(fairness_component="KL")
        assert callable(inst._fairness_component)

    def test_config_resolves_relevance_component(
            self, small_prefs, small_candidates, small_items):
        """config() must resolve relevance_component to a callable."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(relevance_component="SUM")
        assert callable(inst._relevance_component)

    def test_config_resolves_select_item_component(
            self, small_prefs, small_candidates, small_items):
        """config() must resolve select_item_component to a callable."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(select_item_component="SURROGATE")
        assert callable(inst._select_item_component)

    def test_config_stores_environment(self, small_prefs, small_candidates, small_items):
        """config() must populate self.environment."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(list_size=7)
        assert inst.environment["list_size"] == 7

    def test_unknown_distribution_component_raises(
            self, small_prefs, small_candidates, small_items):
        """Unknown distribution acronym must raise NameError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(NameError):
            inst.config(distribution_component="UNKNOWN_DIST")

    def test_unknown_fairness_component_raises(
            self, small_prefs, small_candidates, small_items):
        """Unknown fairness acronym must raise NameError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(NameError):
            inst.config(fairness_component="TOTALLY_WRONG")

    def test_unknown_relevance_component_raises(
            self, small_prefs, small_candidates, small_items):
        """Unknown relevance acronym must raise NameError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(NameError):
            inst.config(relevance_component="BAD_REL")

    def test_unknown_tradeoff_weight_raises(
            self, small_prefs, small_candidates, small_items):
        """Unknown weight acronym must raise NameError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(NameError):
            inst.config(tradeoff_weight_component="NOPE")

    def test_unknown_select_item_raises(
            self, small_prefs, small_candidates, small_items):
        """Unknown selection acronym must raise NameError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(NameError):
            inst.config(select_item_component="GREEDY_MMR")

    def test_default_fairness_component_is_chi_square(
            self, small_prefs, small_candidates, small_items):
        """Default fairness_component must be CHI_SQUARE (regression: was broken 'CHI')."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config()
        assert inst.environment["fairness"] == "CHI_SQUARE"

    @pytest.mark.parametrize("dist", ["CWS", "WPS", "PGD"])
    def test_valid_distribution_components(self, dist, small_prefs, small_candidates, small_items):
        """All documented distribution keys must resolve without error."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(distribution_component=dist)
        assert callable(inst._distribution_component)

    @pytest.mark.parametrize("fair", ["KL", "HELLINGER", "JENSEN_SHANNON", "CHI_SQUARE"])
    def test_valid_fairness_components(self, fair, small_prefs, small_candidates, small_items):
        """Core fairness measures must resolve without error."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(fairness_component=fair)
        assert callable(inst._fairness_component)

    @pytest.mark.parametrize("rel", ["SUM", "NDCG"])
    def test_valid_relevance_components(self, rel, small_prefs, small_candidates, small_items):
        """Documented relevance keys must resolve without error."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(relevance_component=rel)
        assert callable(inst._relevance_component)

    @pytest.mark.parametrize("weight", ["VAR", "STD", "CGR", "TRT", "AMP", "EFF", "C@0.5"])
    def test_valid_weight_components(self, weight, small_prefs, small_candidates, small_items):
        """All documented weight keys must resolve without error."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(tradeoff_weight_component=weight)


# ---------------------------------------------------------------------------
# fit() without config() guard
# ---------------------------------------------------------------------------

class TestLinearCalibrationFitGuard:

    def test_fit_without_config_raises_system_error(
            self, small_prefs, small_candidates, small_items):
        """fit() called before config() must raise SystemError."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        with pytest.raises(SystemError):
            inst.fit()


# ---------------------------------------------------------------------------
# fit() output contract
# ---------------------------------------------------------------------------

class TestLinearCalibrationFitOutput:

    def test_fit_returns_dataframe(self, lc):
        """fit() must return a DataFrame."""
        result = lc.fit()
        assert isinstance(result, pd.DataFrame)

    def test_fit_contains_user_id_column(self, lc):
        """Output DataFrame must contain a USER_ID column."""
        result = lc.fit()
        assert "USER_ID" in result.columns

    def test_fit_contains_item_id_column(self, lc):
        """Output DataFrame must contain an ITEM_ID column."""
        result = lc.fit()
        assert "ITEM_ID" in result.columns

    def test_fit_contains_order_column(self, lc):
        """Output DataFrame must contain an ORDER column."""
        result = lc.fit()
        assert "ORDER" in result.columns

    def test_fit_list_size_rows_per_user(self, lc):
        """Each user must have exactly list_size rows in the output."""
        result = lc.fit()
        for uid, group in result.groupby("USER_ID"):
            assert len(group) == 3, f"User {uid} has {len(group)} rows, expected 3"

    def test_fit_no_duplicate_items_per_user(self, lc):
        """No ITEM_ID must appear twice for the same user."""
        result = lc.fit()
        for uid, group in result.groupby("USER_ID"):
            assert group["ITEM_ID"].nunique() == len(group)

    def test_fit_all_users_present_in_output(self, lc, small_prefs):
        """Every USER_ID from users_preferences must appear in the output."""
        result = lc.fit()
        expected_users = set(small_prefs["USER_ID"].unique())
        output_users = set(result["USER_ID"].unique())
        assert expected_users == output_users

    def test_fit_items_come_from_candidate_pool(self, lc, small_candidates):
        """Every recommended ITEM_ID must come from the user's candidate pool."""
        result = lc.fit()
        for uid, group in result.groupby("USER_ID"):
            pool = set(small_candidates[small_candidates["USER_ID"] == uid]["ITEM_ID"])
            for iid in group["ITEM_ID"]:
                assert iid in pool

    def test_fit_specific_uuids_filters_output(
            self, small_prefs, small_candidates, small_items):
        """When uuids=[1] is passed, output contains only user 1."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(
            distribution_component="CWS",
            fairness_component="KL",
            relevance_component="SUM",
            tradeoff_weight_component="VAR",
            select_item_component="SURROGATE",
            list_size=3,
        )
        result = inst.fit(uuids=[1])
        assert set(result["USER_ID"].unique()) == {1}


# ---------------------------------------------------------------------------
# Idempotent reconfiguration
# ---------------------------------------------------------------------------

class TestLinearCalibrationIdempotentReconfig:

    def test_second_config_overrides_first(self, small_prefs, small_candidates, small_items):
        """A second config() call must not leak state from the first."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(fairness_component="KL", list_size=2)
        inst.config(fairness_component="KL", list_size=3)
        result = inst.fit()
        for _, group in result.groupby("USER_ID"):
            assert len(group) == 3


# ---------------------------------------------------------------------------
# Determinism under fixed seed
# ---------------------------------------------------------------------------

class TestLinearCalibrationDeterminism:

    def test_fit_twice_yields_identical_output(self, small_prefs, small_candidates, small_items):
        """Two fit() calls with identical inputs must return identical DataFrames."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(
            distribution_component="CWS",
            fairness_component="KL",
            relevance_component="SUM",
            tradeoff_weight_component="C@0.5",
            select_item_component="SURROGATE",
            list_size=3,
        )
        r1 = inst.fit()
        r2 = inst.fit()
        tm.assert_frame_equal(
            r1.reset_index(drop=True).sort_values(["USER_ID", "ITEM_ID"]).reset_index(drop=True),
            r2.reset_index(drop=True).sort_values(["USER_ID", "ITEM_ID"]).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Constant lambda boundary behaviour
# ---------------------------------------------------------------------------

class TestLinearCalibrationLambdaBoundary:

    def _make_pure_relevance_instance(self, prefs, cands, items, list_size):
        """λ=0 → pure-relevance mode."""
        inst = LinearCalibration(prefs, cands, items)
        inst.config(
            distribution_component="CWS",
            fairness_component="KL",
            relevance_component="SUM",
            tradeoff_weight_component="C@0.0",
            select_item_component="SURROGATE",
            list_size=list_size,
        )
        return inst

    def test_pure_relevance_lambda_returns_top_k_by_score(
            self, small_prefs, small_candidates, small_items):
        """λ=0 must recommend the top list_size items by TRANSACTION_VALUE for each user."""
        inst = self._make_pure_relevance_instance(
            small_prefs, small_candidates, small_items, list_size=3
        )
        result = inst.fit()
        for uid, group in result.groupby("USER_ID"):
            user_cands = small_candidates[small_candidates["USER_ID"] == uid]
            top3 = set(user_cands.nlargest(3, "TRANSACTION_VALUE")["ITEM_ID"])
            rec_items = set(group["ITEM_ID"])
            assert rec_items == top3, (
                f"User {uid}: expected {top3}, got {rec_items}"
            )


# ---------------------------------------------------------------------------
# Candidate pool smaller than list_size
# ---------------------------------------------------------------------------

class TestLinearCalibrationSmallPool:

    def test_list_size_larger_than_pool_returns_pool_size(self):
        """When list_size > candidate pool size, output length equals pool size."""
        prefs = pd.DataFrame({
            "USER_ID": [1, 1], "ITEM_ID": [10, 20], "TRANSACTION_VALUE": [5.0, 4.0]
        })
        cands = pd.DataFrame({
            "USER_ID": [1, 1], "ITEM_ID": [10, 20], "TRANSACTION_VALUE": [5.0, 4.0]
        })
        items = pd.DataFrame({"ITEM_ID": [10, 20], "GENRES": ["Action", "Drama"]})
        inst = LinearCalibration(prefs, cands, items)
        inst.config(fairness_component="KL", list_size=10)
        result = inst.fit()
        assert len(result[result["USER_ID"] == 1]) == 2

    def test_list_size_one_returns_single_item(self):
        """list_size=1 must return exactly 1 item per user."""
        prefs = pd.DataFrame({
            "USER_ID": [1, 1, 1], "ITEM_ID": [10, 20, 30],
            "TRANSACTION_VALUE": [5.0, 3.0, 1.0]
        })
        cands = pd.DataFrame({
            "USER_ID": [1, 1, 1], "ITEM_ID": [10, 20, 30],
            "TRANSACTION_VALUE": [5.0, 3.0, 1.0]
        })
        items = pd.DataFrame({"ITEM_ID": [10, 20, 30], "GENRES": ["A", "B", "C"]})
        inst = LinearCalibration(prefs, cands, items)
        inst.config(fairness_component="KL", list_size=1)
        result = inst.fit()
        assert len(result[result["USER_ID"] == 1]) == 1


# ---------------------------------------------------------------------------
# Parametric integration: multiple component combinations
# ---------------------------------------------------------------------------

class TestLinearCalibrationComponentCombinations:

    @pytest.mark.parametrize("dist,fair,rel,weight", [
        ("CWS", "KL", "SUM", "VAR"),
        ("CWS", "HELLINGER", "SUM", "STD"),
        ("CWS", "JENSEN_SHANNON", "SUM", "C@0.5"),
        ("PGD", "CHI_SQUARE", "NDCG", "CGR"),
    ])
    def test_full_pipeline_runs_and_returns_correct_schema(
            self, dist, fair, rel, weight,
            small_prefs, small_candidates, small_items):
        """Full construct→config→fit pipeline must succeed for various component combos."""
        inst = LinearCalibration(small_prefs, small_candidates, small_items)
        inst.config(
            distribution_component=dist,
            fairness_component=fair,
            relevance_component=rel,
            tradeoff_weight_component=weight,
            select_item_component="SURROGATE",
            list_size=3,
        )
        result = inst.fit()
        assert isinstance(result, pd.DataFrame)
        assert {"USER_ID", "ITEM_ID", "ORDER"}.issubset(result.columns)
        for _, group in result.groupby("USER_ID"):
            assert len(group) <= 3
