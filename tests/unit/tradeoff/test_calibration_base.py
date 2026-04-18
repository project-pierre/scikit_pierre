"""
Unit tests for CalibrationBase trade-off balance helpers.

Verifies the additive (similarity) and subtractive (divergence) formulas
and the measure-router that selects between them.
"""
import pytest

from scikit_pierre.tradeoff.calibration import CalibrationBase
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal fixtures to construct CalibrationBase (abstract via subclass)
# ---------------------------------------------------------------------------

@pytest.fixture
def base_instance():
    """Return a CalibrationBase-derived object using LinearCalibration."""
    from scikit_pierre.tradeoff.calibration import LinearCalibration
    prefs = pd.DataFrame({
        "USER_ID": [1, 1], "ITEM_ID": [10, 20], "TRANSACTION_VALUE": [4.0, 3.0]
    })
    cands = pd.DataFrame({
        "USER_ID": [1], "ITEM_ID": [20], "TRANSACTION_VALUE": [3.5]
    })
    items = pd.DataFrame({"ITEM_ID": [10, 20], "GENRES": ["Action", "Drama"]})
    return LinearCalibration(prefs, cands, items)


# ---------------------------------------------------------------------------
# _tradeoff_sim
# ---------------------------------------------------------------------------

class TestTradeoffSim:

    def test_zero_lambda_returns_only_relevance(self):
        """λ=0 → utility equals relevance_value."""
        result = CalibrationBase._tradeoff_sim(0.0, 5.0, 2.0)
        assert result == pytest.approx(5.0)

    def test_one_lambda_returns_only_fairness(self):
        """λ=1 → utility equals fairness_value."""
        result = CalibrationBase._tradeoff_sim(1.0, 5.0, 2.0)
        assert result == pytest.approx(2.0)

    def test_half_lambda_averages_both(self):
        """λ=0.5 → utility = 0.5*rel + 0.5*fair."""
        result = CalibrationBase._tradeoff_sim(0.5, 4.0, 2.0)
        assert result == pytest.approx(3.0)

    def test_formula_additive(self):
        """(1-λ)*rel + λ*fair for arbitrary values."""
        result = CalibrationBase._tradeoff_sim(0.3, 10.0, 5.0)
        assert result == pytest.approx(0.7 * 10.0 + 0.3 * 5.0)

    def test_returns_float(self):
        result = CalibrationBase._tradeoff_sim(0.4, 3.0, 1.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _tradeoff_div
# ---------------------------------------------------------------------------

class TestTradeoffDiv:

    def test_zero_lambda_returns_only_relevance(self):
        """λ=0 → utility equals relevance_value."""
        result = CalibrationBase._tradeoff_div(0.0, 5.0, 2.0)
        assert result == pytest.approx(5.0)

    def test_one_lambda_subtracts_fairness(self):
        """λ=1 → utility = -fairness_value."""
        result = CalibrationBase._tradeoff_div(1.0, 0.0, 3.0)
        assert result == pytest.approx(-3.0)

    def test_half_lambda(self):
        """λ=0.5 → utility = 0.5*rel - 0.5*fair."""
        result = CalibrationBase._tradeoff_div(0.5, 4.0, 2.0)
        assert result == pytest.approx(1.0)

    def test_formula_subtractive(self):
        """(1-λ)*rel - λ*fair for arbitrary values."""
        result = CalibrationBase._tradeoff_div(0.3, 10.0, 5.0)
        assert result == pytest.approx(0.7 * 10.0 - 0.3 * 5.0)

    def test_returns_float(self):
        result = CalibrationBase._tradeoff_div(0.2, 3.0, 1.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _tradeoff_funcs router
# ---------------------------------------------------------------------------

class TestTradeoffFuncsRouter:

    @pytest.mark.parametrize("sim_measure", [
        "INTERSECTION_SIM", "CZEKANOWSKI_SIM", "MOTYKA_SIM",
        "COSINE", "FIDELITY", "RUZICKA",
    ])
    def test_similarity_measures_route_to_sim(self, base_instance, sim_measure):
        """Similarity measures must route to _tradeoff_sim."""
        func = base_instance._tradeoff_funcs(sim_measure)
        assert func is CalibrationBase._tradeoff_sim

    @pytest.mark.parametrize("div_measure", [
        "KL", "JS", "HELLINGER", "CHI_SQUARE", "EUCLIDEAN", "KL",
    ])
    def test_divergence_measures_route_to_div(self, base_instance, div_measure):
        """Divergence measures must route to _tradeoff_div."""
        func = base_instance._tradeoff_funcs(div_measure)
        assert func is CalibrationBase._tradeoff_div

    def test_router_is_case_insensitive_for_sim(self, base_instance):
        """Routing check uses .upper() so lowercase input still works."""
        func = base_instance._tradeoff_funcs("cosine")
        assert func is CalibrationBase._tradeoff_sim
