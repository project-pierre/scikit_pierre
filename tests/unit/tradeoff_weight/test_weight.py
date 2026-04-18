"""
Unit tests for scikit_pierre.tradeoff_weight.weight.

Covers: genre_count, norm_var, norm_std, trust, amplitude, efficiency, mitigation.
"""
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scikit_pierre.tradeoff_weight.weight import (
    amplitude,
    efficiency,
    genre_count,
    mitigation,
    norm_std,
    norm_var,
    trust,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

UNIFORM_4 = [0.25, 0.25, 0.25, 0.25]
DELTA_4 = [1.0, 0.0, 0.0, 0.0]
ALL_ZERO_4 = [0.0, 0.0, 0.0, 0.0]
ALL_ONE_4 = [1.0, 1.0, 1.0, 1.0]
MIXED_6 = [0.5, 0.2, 0.1, 0.0, 0.0, 0.0]
MIXED_7 = [0.269, 0.192, 0.076, 0.384, 0.0, 0.0, 0.076]


@pytest.fixture
def uniform():
    return list(UNIFORM_4)


@pytest.fixture
def delta():
    return list(DELTA_4)


@pytest.fixture
def all_zero():
    return list(ALL_ZERO_4)


# ---------------------------------------------------------------------------
# Hypothesis strategy: valid probability distribution (values in [0,1])
# ---------------------------------------------------------------------------

@st.composite
def prob_dist(draw, min_size=1, max_size=20):
    """Draw a non-empty list of non-negative floats that sum to 1."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    raw = draw(st.lists(st.floats(min_value=0.0, max_value=1.0,
                                   allow_nan=False, allow_infinity=False),
                        min_size=n, max_size=n))
    total = sum(raw)
    if total == 0.0:
        raw = [1.0 / n] * n
        total = 1.0
    return [v / total for v in raw]


# ===========================================================================
# genre_count
# ===========================================================================

class TestGenreCount:

    @pytest.mark.parametrize("dist_vec,expected", [
        ([0.5, 0.2, 0.1, 0.0, 0.0, 0.0], 3 / 6),
        ([0.269, 0.192, 0.076, 0.384, 0.0, 0.0, 0.076], 5 / 7),
        ([0.0, 0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0], 1.0),
        ([0.25, 0.25, 0.25, 0.25], 1.0),
        ([1.0, 0.0, 0.0, 0.0], 0.25),
        ([0.5], 1.0),
        ([0.0], 0.0),
        ([0.1, 0.0, 0.9], 2 / 3),
        ([0.0, 0.0, 0.0, 0.0, 1.0], 0.2),
    ])
    def test_genre_count_golden_values(self, dist_vec, expected):
        """Pin hand-computed genre count ratios."""
        assert genre_count(dist_vec) == pytest.approx(expected)

    def test_genre_count_bounded_min(self, all_zero):
        """All-zero distribution yields lambda = 0.0."""
        assert genre_count(all_zero) == 0.0

    def test_genre_count_bounded_max(self):
        """All-active distribution yields lambda = 1.0."""
        assert genre_count([0.1, 0.2, 0.3, 0.4]) == 1.0

    def test_genre_count_in_unit_interval(self):
        """Result is always in [0, 1]."""
        for dist in [MIXED_6, MIXED_7, UNIFORM_4, DELTA_4, ALL_ZERO_4, ALL_ONE_4]:
            lam = genre_count(dist)
            assert 0.0 <= lam <= 1.0

    def test_genre_count_deterministic(self):
        """Identical inputs produce identical outputs."""
        dist = [0.5, 0.3, 0.2]
        assert genre_count(dist) == genre_count(dist)

    def test_genre_count_invariant_to_order(self):
        """Reordering the non-zero elements must not change the count."""
        assert genre_count([0.5, 0.3, 0.2, 0.0]) == genre_count([0.0, 0.2, 0.3, 0.5])

    def test_genre_count_returns_float(self):
        """Return type is float."""
        assert isinstance(genre_count(UNIFORM_4), float)

    def test_genre_count_finite(self):
        """Result is never NaN or inf."""
        lam = genre_count(MIXED_6)
        assert math.isfinite(lam)

    def test_genre_count_monotone_more_active(self):
        """Adding an active genre increases (or keeps equal) the count ratio."""
        base = [0.5, 0.5, 0.0, 0.0]
        extended = [0.4, 0.4, 0.2, 0.0]
        assert genre_count(extended) >= genre_count(base)

    @given(prob_dist())
    @settings(max_examples=300)
    def test_genre_count_hypothesis_bounded(self, dist):
        """Property: genre_count is in [0, 1] for any valid prob distribution."""
        lam = genre_count(dist)
        assert 0.0 <= lam <= 1.0
        assert math.isfinite(lam)


# ===========================================================================
# norm_var
# ===========================================================================

class TestNormVar:

    def test_norm_var_uniform_is_one(self, uniform):
        """Uniform distribution has zero variance → norm_var = 1.0."""
        assert norm_var(uniform) == pytest.approx(1.0)

    def test_norm_var_all_equal_arbitrary(self):
        """Any constant vector has zero variance → norm_var = 1.0."""
        assert norm_var([0.5, 0.5, 0.5, 0.5]) == pytest.approx(1.0)

    def test_norm_var_delta_golden(self, delta):
        """Hand-computed: delta [1,0,0,0] → var=3/16=0.1875, result=0.8125."""
        expected = 1 - 0.1875
        assert norm_var(delta) == pytest.approx(expected)

    @pytest.mark.parametrize("dist_vec", [
        UNIFORM_4, DELTA_4, MIXED_6, MIXED_7, ALL_ZERO_4, ALL_ONE_4,
    ])
    def test_norm_var_in_unit_interval(self, dist_vec):
        """norm_var is in [0, 1] for valid probability distributions."""
        lam = norm_var(list(dist_vec))
        assert 0.0 <= lam <= 1.0

    def test_norm_var_deterministic(self):
        """Same input → same output."""
        dist = [0.4, 0.3, 0.2, 0.1]
        assert norm_var(dist) == norm_var(dist)

    def test_norm_var_returns_float(self):
        assert isinstance(norm_var(UNIFORM_4), float)

    def test_norm_var_finite(self):
        assert math.isfinite(norm_var(MIXED_6))

    def test_norm_var_uniform_greater_than_concentrated(self, uniform, delta):
        """Uniform dist has zero variance (norm_var=1.0); delta has positive variance."""
        assert norm_var(uniform) > norm_var(delta)

    def test_norm_var_single_element(self):
        """Single-element vector has zero variance → 1.0."""
        assert norm_var([0.7]) == pytest.approx(1.0)

    def test_norm_var_invariant_to_order(self):
        """Result does not depend on element order."""
        dist = [0.5, 0.3, 0.2]
        assert norm_var(dist) == pytest.approx(norm_var([0.2, 0.5, 0.3]))

    def test_norm_var_two_extreme_elements(self):
        """[0, 1]: mean=0.5, var=0.25 → norm_var=0.75."""
        assert norm_var([0.0, 1.0]) == pytest.approx(0.75)

    @given(prob_dist())
    @settings(max_examples=300)
    def test_norm_var_hypothesis_bounded(self, dist):
        """norm_var ∈ [0,1] and finite for any valid prob distribution."""
        lam = norm_var(dist)
        assert 0.0 <= lam <= 1.0
        assert math.isfinite(lam)


# ===========================================================================
# norm_std
# ===========================================================================

class TestNormStd:

    def test_norm_std_uniform_is_one(self, uniform):
        """std of uniform distribution is 0 → norm_std = 1.0."""
        assert norm_std(uniform) == pytest.approx(1.0)

    def test_norm_std_delta_golden(self, delta):
        """Hand-computed: delta [1,0,0,0] → std=sqrt(0.1875), result=1-sqrt(0.1875)."""
        expected = 1 - math.sqrt(0.1875)
        assert norm_std(delta) == pytest.approx(expected)

    @pytest.mark.parametrize("dist_vec", [
        UNIFORM_4, DELTA_4, MIXED_6, MIXED_7,
    ])
    def test_norm_std_in_unit_interval(self, dist_vec):
        """norm_std is in [0, 1] for valid probability distributions."""
        lam = norm_std(list(dist_vec))
        assert 0.0 <= lam <= 1.0

    def test_norm_std_deterministic(self):
        dist = [0.4, 0.3, 0.2, 0.1]
        assert norm_std(dist) == norm_std(dist)

    def test_norm_std_returns_float(self):
        assert isinstance(norm_std(UNIFORM_4), float)

    def test_norm_std_finite(self):
        assert math.isfinite(norm_std(MIXED_6))

    def test_norm_std_uniform_greater_than_concentrated(self, uniform, delta):
        """Uniform → std=0 (norm_std=1.0); delta → std>0 (norm_std<1.0)."""
        assert norm_std(uniform) > norm_std(delta)

    def test_norm_std_single_element(self):
        """Single-element vector has std=0 → 1.0."""
        assert norm_std([0.8]) == pytest.approx(1.0)

    def test_norm_std_invariant_to_order(self):
        dist = [0.5, 0.3, 0.2]
        assert norm_std(dist) == pytest.approx(norm_std([0.2, 0.5, 0.3]))

    def test_norm_std_two_extreme_elements(self):
        """[0, 1]: std=0.5 → norm_std=0.5."""
        assert norm_std([0.0, 1.0]) == pytest.approx(0.5)

    def test_norm_std_less_than_norm_var_for_spread_dist(self):
        """std >= var for values in [0,1], so norm_std <= norm_var for spread dists."""
        dist = [0.0, 1.0]
        assert norm_std(dist) <= norm_var(dist)

    def test_norm_std_all_equal_arbitrary(self):
        assert norm_std([0.3, 0.3, 0.3]) == pytest.approx(1.0)

    @given(prob_dist())
    @settings(max_examples=300)
    def test_norm_std_hypothesis_bounded(self, dist):
        """norm_std ∈ [0,1] and finite for any valid prob distribution."""
        lam = norm_std(dist)
        assert 0.0 <= lam <= 1.0
        assert math.isfinite(lam)


# ===========================================================================
# trust
# ===========================================================================

class TestTrust:

    @pytest.mark.parametrize("dist_vec,expected", [
        ([0.25, 0.25, 0.25, 0.25], 0.25),
        ([0.5, 0.5], 0.5),
        ([1.0], 1.0),
        ([0.0], 0.0),
        ([0.0, 0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0], 1.0),
        ([0.1, 0.2, 0.3, 0.4], 0.25),
        ([0.5, 0.2, 0.1, 0.0, 0.0, 0.0], 0.8 / 6),
        ([0.6, 0.4], 0.5),
        ([1.0, 0.0], 0.5),
    ])
    def test_trust_golden_values(self, dist_vec, expected):
        """Pin hand-computed arithmetic mean values."""
        assert trust(dist_vec) == pytest.approx(expected)

    def test_trust_returns_float(self):
        assert isinstance(trust(UNIFORM_4), float)

    def test_trust_deterministic(self):
        dist = [0.3, 0.3, 0.4]
        assert trust(dist) == trust(dist)

    def test_trust_finite(self):
        assert math.isfinite(trust(MIXED_6))

    def test_trust_invariant_to_order(self):
        assert trust([0.5, 0.3, 0.2]) == pytest.approx(trust([0.2, 0.5, 0.3]))

    def test_trust_positive_for_positive_input(self):
        assert trust([0.1, 0.2, 0.7]) > 0.0

    def test_trust_single_element(self):
        assert trust([0.7]) == pytest.approx(0.7)

    def test_trust_monotone_with_scale(self):
        """Scaling all values up increases the mean."""
        assert trust([0.2, 0.4, 0.6]) > trust([0.1, 0.2, 0.3])

    @given(prob_dist())
    @settings(max_examples=300)
    def test_trust_hypothesis_positive(self, dist):
        """Mean of a non-negative distribution is non-negative."""
        lam = trust(dist)
        assert lam >= 0.0
        assert math.isfinite(lam)


# ===========================================================================
# amplitude
# ===========================================================================

class TestAmplitude:

    def test_amplitude_uniform_is_one(self, uniform):
        """All identical values → all pairwise distances = 0 → amplitude = 1.0."""
        assert amplitude(uniform) == pytest.approx(1.0)

    def test_amplitude_all_equal_arbitrary(self):
        assert amplitude([0.3, 0.3, 0.3]) == pytest.approx(1.0)

    def test_amplitude_single_element(self):
        """Single-element vector has no pairs → magnitude = 0 → amplitude = 1.0."""
        assert amplitude([0.7]) == pytest.approx(1.0)

    def test_amplitude_two_extreme_golden(self):
        """[0, 1]: pairwise distances = [1, 1], magnitude = 2, n^2 = 4 → 1 - 0.5 = 0.5."""
        assert amplitude([0.0, 1.0]) == pytest.approx(0.5)

    def test_amplitude_four_element_golden(self):
        """[0.5, 0.5, 0.0, 0.0]: hand-computed magnitude=4, n^2=16 → 0.75."""
        assert amplitude([0.5, 0.5, 0.0, 0.0]) == pytest.approx(0.75)

    def test_amplitude_three_element_golden(self):
        """[0.5, 0.3, 0.2]: magnitude=1.2, n^2=9 → 1 - 1.2/9."""
        expected = 1 - 1.2 / 9
        assert amplitude([0.5, 0.3, 0.2]) == pytest.approx(expected)

    def test_amplitude_deterministic(self):
        dist = [0.4, 0.3, 0.2, 0.1]
        assert amplitude(dist) == amplitude(dist)

    def test_amplitude_returns_float(self):
        assert isinstance(amplitude(UNIFORM_4), float)

    def test_amplitude_finite(self):
        assert math.isfinite(amplitude(MIXED_6))

    def test_amplitude_in_unit_interval_for_standard_dists(self):
        for dist in [UNIFORM_4, DELTA_4, MIXED_6, MIXED_7]:
            lam = amplitude(list(dist))
            assert 0.0 <= lam <= 1.0

    def test_amplitude_invariant_to_order(self):
        """Pairwise distances are symmetric, so order must not matter."""
        dist = [0.5, 0.3, 0.2]
        assert amplitude(dist) == pytest.approx(amplitude([0.2, 0.5, 0.3]))

    def test_amplitude_concentrated_lower_than_uniform(self):
        """Concentrated distribution → large pairwise spread → amplitude < 1 (< uniform)."""
        concentrated = [0.9, 0.05, 0.05]
        uniform = [1 / 3, 1 / 3, 1 / 3]
        assert amplitude(concentrated) < amplitude(uniform)

    @given(prob_dist())
    @settings(max_examples=300)
    def test_amplitude_hypothesis_bounded(self, dist):
        """amplitude ∈ [0,1] and finite for valid prob distributions."""
        lam = amplitude(dist)
        assert 0.0 <= lam <= 1.0
        assert math.isfinite(lam)


# ===========================================================================
# efficiency
# ===========================================================================

class TestEfficiency:

    def test_efficiency_uniform_is_zero(self, uniform):
        """Uniform distribution has zero variance → efficiency = 0.0."""
        assert efficiency(uniform) == pytest.approx(0.0)

    def test_efficiency_constant_vector_is_zero(self):
        assert efficiency([0.5, 0.5]) == pytest.approx(0.0)

    def test_efficiency_golden_two_element(self):
        """[0.4, 0.6]: mean=0.5, var=0.01, mean^2=0.25 → 0.04."""
        assert efficiency([0.4, 0.6]) == pytest.approx(0.04)

    def test_efficiency_golden_more_spread(self):
        """[0.2, 0.8]: mean=0.5, var=0.09, mean^2=0.25 → 0.36."""
        assert efficiency([0.2, 0.8]) == pytest.approx(0.36)

    def test_efficiency_monotone_with_spread(self):
        """More spread distribution → higher efficiency."""
        assert efficiency([0.2, 0.8]) > efficiency([0.4, 0.6])

    def test_efficiency_deterministic(self):
        dist = [0.4, 0.3, 0.2, 0.1]
        assert efficiency(dist) == efficiency(dist)

    def test_efficiency_returns_float(self):
        assert isinstance(efficiency(UNIFORM_4), float)

    def test_efficiency_finite(self):
        assert math.isfinite(efficiency(MIXED_6))

    def test_efficiency_single_element(self):
        """Single-element vector: variance=0 → efficiency=0.0."""
        assert efficiency([0.5]) == pytest.approx(0.0)

    def test_efficiency_invariant_to_order(self):
        """Variance and mean are order-invariant."""
        dist = [0.5, 0.3, 0.2]
        assert efficiency(dist) == pytest.approx(efficiency([0.2, 0.5, 0.3]))

    def test_efficiency_non_negative_for_valid_dists(self):
        """Efficiency = variance / mean^2, both numerator and denominator non-negative."""
        for dist in [UNIFORM_4, DELTA_4, MIXED_6]:
            assert efficiency(list(dist)) >= 0.0

    def test_efficiency_golden_three_elements(self):
        """[0.1, 0.4, 0.5]: mean=1/3, var=((0.1-1/3)^2+(0.4-1/3)^2+(0.5-1/3)^2)/3."""
        dist = [0.1, 0.4, 0.5]
        mean = sum(dist) / 3
        var = sum((x - mean) ** 2 for x in dist) / 3
        expected = var / mean ** 2
        assert efficiency(dist) == pytest.approx(expected)

    @given(prob_dist(min_size=2))
    @settings(max_examples=300)
    def test_efficiency_hypothesis_non_negative(self, dist):
        """efficiency >= 0 and finite for any valid prob distribution."""
        lam = efficiency(dist)
        assert lam >= 0.0
        assert math.isfinite(lam)


# ===========================================================================
# mitigation
# ===========================================================================

class TestMitigation:

    def test_mitigation_identical_dists_golden(self):
        """dist_vec=[1.0], target=cand=[1.0]: ndcg=1, jsf=1 → result=0.5."""
        result = mitigation([1.0], [1.0], [1.0])
        assert result == pytest.approx(0.5)

    def test_mitigation_identical_dists_uniform(self):
        """When target==cand, JS=0, jsf=1; ndcg on sorted scores = 1.0 → result=0.5."""
        result = mitigation([0.8, 0.6, 0.4], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2])
        assert result == pytest.approx(0.5)

    def test_mitigation_returns_float(self):
        assert isinstance(mitigation([1.0], [1.0], [1.0]), float)

    def test_mitigation_finite(self):
        result = mitigation([0.8, 0.6, 0.4], [0.5, 0.5], [0.4, 0.6])
        assert math.isfinite(result)

    def test_mitigation_deterministic(self):
        args = ([0.8, 0.6], [0.5, 0.5], [0.4, 0.6])
        assert mitigation(*args) == mitigation(*args)

    def test_mitigation_result_positive(self):
        """Harmonic mean of two positive values is positive."""
        result = mitigation([1.0, 0.5], [0.6, 0.4], [0.5, 0.5])
        assert result > 0.0

    def test_mitigation_in_unit_interval(self):
        """Result must lie in (0, 1]."""
        result = mitigation([1.0, 0.8, 0.5], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2])
        assert 0.0 < result <= 1.0

    def test_mitigation_higher_relevance_increases_lambda(self):
        """Keeping distributions fixed, scores closer to ideal → higher ndcg → higher λ.
        Ideal order (desc) vs reverse order: ndcg differs; jsf same."""
        target = [0.5, 0.5]
        cand = [0.5, 0.5]
        ideal = mitigation([1.0, 0.5], target, cand)
        reversed_scores = mitigation([0.5, 1.0], target, cand)
        assert ideal >= reversed_scores

    def test_mitigation_closer_cand_to_target_increases_lambda(self):
        """Same dist_vec; cand closer to target → higher jsf → higher λ."""
        dist_vec = [1.0, 0.8]
        target = [0.6, 0.4]
        close_cand = [0.59, 0.41]
        far_cand = [0.1, 0.9]
        result_close = mitigation(dist_vec, target, close_cand)
        result_far = mitigation(dist_vec, target, far_cand)
        assert result_close > result_far

    def test_mitigation_zero_scores_gives_zero(self):
        """All-zero scores → ndcg=0 → numerator=0 → result=0.0."""
        result = mitigation([0.0, 0.0], [0.5, 0.5], [0.5, 0.5])
        assert result == pytest.approx(0.0)

    def test_mitigation_perfect_relevance_and_calibration(self):
        """ndcg=1 and jsf=1 → harmonic mean = 0.5."""
        dist_vec = [1.0]
        result = mitigation(dist_vec, [1.0], [1.0])
        assert result == pytest.approx(0.5)

    def test_mitigation_single_item_golden(self):
        """Single-item list: ndcg=1, jsf=1-JS(target,cand) where target!=cand.
        result = (1 * jsf) / (1 + jsf) = jsf / (1 + jsf)."""
        from scikit_pierre.measures.shannon import jensen_shannon
        target = [0.7, 0.3]
        cand = [0.4, 0.6]
        jsf = 1.0 - jensen_shannon(target, cand)
        expected = jsf / (1.0 + jsf)
        result = mitigation([1.0], target, cand)
        assert result == pytest.approx(expected, rel=1e-6)
