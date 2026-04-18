"""
Unit tests for scikit_pierre.tradeoff_weight.accessible (factory dispatcher).
"""
import pytest

from scikit_pierre.tradeoff_weight import weight
from scikit_pierre.tradeoff_weight.accessible import tradeoff_weights_funcs


class TestConstantLambda:

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_constant_returns_exact_float(self, value):
        """C@<v> must return exactly float(v), for all values in [0, 1]."""
        result = tradeoff_weights_funcs(f"C@{value}")
        assert result == pytest.approx(value)
        assert isinstance(result, float)

    def test_constant_zero(self):
        """C@0.0 → pure-relevance boundary: lambda = 0.0."""
        assert tradeoff_weights_funcs("C@0.0") == 0.0

    def test_constant_one(self):
        """C@1.0 → pure-calibration boundary: lambda = 1.0."""
        assert tradeoff_weights_funcs("C@1.0") == 1.0

    def test_constant_parses_scientific_notation(self):
        """C@1e-2 must parse correctly as 0.01."""
        assert tradeoff_weights_funcs("C@1e-2") == pytest.approx(0.01)

    def test_constant_splits_on_at_sign(self):
        """Only the portion after '@' is parsed."""
        assert tradeoff_weights_funcs("C@0.333") == pytest.approx(0.333)


class TestCallableStrategies:

    @pytest.mark.parametrize("key,expected_func", [
        ("CGR", weight.genre_count),
        ("VAR", weight.norm_var),
        ("STD", weight.norm_std),
        ("TRT", weight.trust),
        ("AMP", weight.amplitude),
        ("EFF", weight.efficiency),
        ("MIT", weight.mitigation),
    ])
    def test_returns_correct_function(self, key, expected_func):
        """Each acronym must resolve to the documented callable."""
        assert tradeoff_weights_funcs(key) is expected_func

    @pytest.mark.parametrize("key", ["CGR", "VAR", "STD", "TRT", "AMP", "EFF", "MIT"])
    def test_returned_value_is_callable(self, key):
        func = tradeoff_weights_funcs(key)
        assert callable(func)

    @pytest.mark.parametrize("key", ["CGR", "VAR", "STD", "TRT", "AMP", "EFF"])
    def test_callable_produces_float_on_simple_input(self, key):
        """Every single-arg strategy must accept a list and return a float."""
        func = tradeoff_weights_funcs(key)
        result = func([0.5, 0.3, 0.2])
        assert isinstance(result, float)

    def test_mit_callable_produces_float(self):
        func = tradeoff_weights_funcs("MIT")
        result = func([1.0, 0.5], [0.6, 0.4], [0.5, 0.5])
        assert isinstance(result, float)


class TestInvalidInput:

    @pytest.mark.parametrize("bad_key", [
        "UNKNOWN", "cgr", "var", "", "LAMBDA", "RANDOM",
    ])
    def test_unknown_key_raises_name_error(self, bad_key):
        """Any unrecognised key must raise NameError."""
        with pytest.raises(NameError):
            tradeoff_weights_funcs(bad_key)

    def test_error_message_contains_bad_key(self):
        bad = "TOTALLY_UNKNOWN_XYZ"
        with pytest.raises(NameError, match=bad):
            tradeoff_weights_funcs(bad)

    def test_lowercase_cgr_raises(self):
        with pytest.raises(NameError):
            tradeoff_weights_funcs("cgr")

    def test_bare_c_at_raises(self):
        """'C@' with no numeric suffix raises ValueError from float()."""
        with pytest.raises((NameError, ValueError)):
            tradeoff_weights_funcs("C@")
