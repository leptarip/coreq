# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pin the log-domain machinery under the importance-sampling estimator.

`test_estimators.py` covers the composed results -- the estimate, the quality
gates, a Gaussian-tail sanity check. What it does not touch is the numerics
those results are assembled from, or the nominal baseline that every reported
variance-reduction factor is a ratio against. Both are covered here:

  * `_log_binomial_cdf` and `clopper_pearson_upper` against the equation that
    defines a Clopper-Pearson limit, in a regime where a naive CDF underflows
    to zero, and against the *other* CP implementation in this repository;
  * `_log_scale_delta_interval` against the multiplicative symmetry that
    characterises a delta-method interval built in log space;
  * `_representable_exp` and `_validate_inputs` against the overflow, underflow
    and malformed-input cases they exist to absorb;
  * `naive_monte_carlo_summary`, the denominator of every IS gain claim.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest
from scipy.stats import binom

from source.design_exploration.commons.scenario_labeling import (
    cp_upper_bound as cp_upper_bound_via_beta_quantile,
)
from source.design_exploration.rare_events.estimators import (
    _LOG_MAX_FLOAT,
    _LOG_MIN_FLOAT,
    _log_binomial_cdf,
    _log_scale_delta_interval,
    _representable_exp,
    _validate_inputs,
    clopper_pearson_upper,
    importance_sampling_estimate,
    naive_monte_carlo_summary,
    required_zero_failure_budget,
    severity_curve,
)


# --------------------------------------------------------------------------
# Representable exponentiation
# --------------------------------------------------------------------------


class TestRepresentableExp:
    """`None` means "cannot be written as a float"; `0.0` means "genuinely
    zero, or below the smallest normal". Conflating the two would turn an
    overflow into a silent zero estimate."""

    def test_a_log_of_negative_infinity_is_an_exact_zero(self):
        assert _representable_exp(-math.inf) == 0.0

    @pytest.mark.parametrize("value", [math.inf, float("nan")])
    def test_non_finite_input_is_unrepresentable(self, value):
        assert _representable_exp(value) is None

    def test_the_largest_representable_exponent_still_converts(self):
        assert _representable_exp(_LOG_MAX_FLOAT) == pytest.approx(1.7976931348622732e308)

    @pytest.mark.parametrize("excess", [1e-9, 0.5, 100.0])
    def test_anything_above_it_is_unrepresentable(self, excess):
        assert _representable_exp(_LOG_MAX_FLOAT + excess) is None

    def test_the_smallest_normal_exponent_still_converts(self):
        assert _representable_exp(_LOG_MIN_FLOAT) > 0.0

    @pytest.mark.parametrize("shortfall", [1e-9, 10.0, 100.0])
    def test_anything_below_it_flushes_to_zero(self, shortfall):
        assert _representable_exp(_LOG_MIN_FLOAT - shortfall) == 0.0

    def test_subnormal_results_are_flushed_rather_than_returned(self):
        """`exp(-709)` is a representable subnormal, but the guard is written
        against the smallest *normal*, so it returns zero. Pinned because it
        sets the precision floor for every reported estimate."""
        assert math.exp(-709.0) > 0.0
        assert _representable_exp(-709.0) == 0.0

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 100.0, -100.0])
    def test_ordinary_values_round_trip(self, value):
        assert _representable_exp(value) == pytest.approx(math.exp(value))

    def test_it_is_monotone_where_it_returns_a_number(self):
        values = [_representable_exp(v) for v in (-700.0, -10.0, 0.0, 10.0, 700.0)]

        assert values == sorted(values)


# --------------------------------------------------------------------------
# Input guards
# --------------------------------------------------------------------------


class TestValidateInputs:

    def test_matched_non_empty_inputs_are_accepted(self):
        assert _validate_inputs([0.0, -1.0], [True, False]) is None

    def test_empty_inputs_are_refused(self):
        with pytest.raises(ValueError, match="non-empty and have equal length"):
            _validate_inputs([], [])

    @pytest.mark.parametrize(
        "log_weights,indicators",
        [([0.0], [True, False]), ([0.0, 1.0], [True]), ([0.0], [])],
    )
    def test_mismatched_lengths_are_refused(self, log_weights, indicators):
        with pytest.raises(ValueError, match="non-empty and have equal length"):
            _validate_inputs(log_weights, indicators)

    @pytest.mark.parametrize("bad", [float("nan"), math.inf])
    def test_nan_and_positive_infinity_are_refused(self, bad):
        with pytest.raises(ValueError, match="not NaN or"):
            _validate_inputs([0.0, bad], [True, False])

    def test_negative_infinity_is_a_legitimate_weight(self):
        """A draw outside the base support has zero likelihood ratio, which is
        `-inf` in log space -- that is data, not corruption."""
        assert _validate_inputs([-math.inf, 0.0], [True, False]) is None

    def test_the_public_estimator_enforces_the_same_guards(self):
        with pytest.raises(ValueError, match="non-empty and have equal length"):
            importance_sampling_estimate([], [])
        with pytest.raises(ValueError, match="not NaN or"):
            importance_sampling_estimate([float("nan")], [True])

    @pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
    def test_the_public_estimator_rejects_a_degenerate_confidence_level(self, delta):
        with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
            importance_sampling_estimate([0.0], [True], delta=delta)


# --------------------------------------------------------------------------
# Log-domain binomial CDF
# --------------------------------------------------------------------------


class TestLogBinomialCdf:

    @pytest.mark.parametrize(
        "k,n,p",
        [(0, 10, 0.5), (3, 20, 0.3), (0, 1, 0.5), (15, 100, 0.1), (2, 50, 0.05), (7, 7, 0.9)],
    )
    def test_it_matches_the_binomial_cdf(self, k, n, p):
        assert _log_binomial_cdf(k, n, p) == pytest.approx(math.log(binom.cdf(k, n, p)))

    def test_it_survives_a_tail_where_the_plain_cdf_underflows(self):
        """`P(X = 0 | n = 10000, p = 0.5)` is 2**-10000, which is exactly zero
        as a float. Working in logs is the entire point of this helper."""
        assert binom.cdf(0, 10000, 0.5) == 0.0

        assert _log_binomial_cdf(0, 10000, 0.5) == pytest.approx(10000 * math.log(0.5))

    @pytest.mark.parametrize("n", [1000, 10000, 100000])
    def test_the_far_tail_stays_finite_at_scale(self, n):
        assert math.isfinite(_log_binomial_cdf(0, n, 0.5))

    def test_it_is_non_decreasing_in_the_observed_count(self):
        values = [_log_binomial_cdf(k, 30, 0.3) for k in range(10)]

        assert values == sorted(values)

    def test_it_is_non_increasing_in_the_success_probability(self):
        values = [_log_binomial_cdf(5, 30, p) for p in (0.1, 0.2, 0.3, 0.5)]

        assert values == sorted(values, reverse=True)

    def test_the_full_count_is_certain(self):
        assert _log_binomial_cdf(10, 10, 0.4) == pytest.approx(0.0)

    @pytest.mark.parametrize("p", [0.0, -0.5, -1.0])
    def test_a_non_positive_rate_makes_every_count_certain(self, p):
        """With p <= 0 the variate is always zero, so `P(X <= k) = 1`."""
        assert _log_binomial_cdf(3, 10, p) == 0.0

    def test_a_certain_rate_puts_all_mass_at_the_full_count(self):
        assert _log_binomial_cdf(10, 10, 1.0) == 0.0
        assert _log_binomial_cdf(3, 10, 1.0) == -math.inf


# --------------------------------------------------------------------------
# Clopper-Pearson upper limit
# --------------------------------------------------------------------------


class TestClopperPearsonUpper:

    @pytest.mark.parametrize("k,n", [(0, 0), (-1, 10), (11, 10), (1, 0)])
    def test_impossible_counts_are_refused(self, k, n):
        with pytest.raises(ValueError, match="Require n >= 1"):
            clopper_pearson_upper(k, n, 0.05)

    @pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 2.0])
    def test_a_degenerate_confidence_level_is_refused(self, delta):
        with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
            clopper_pearson_upper(0, 10, delta)

    @pytest.mark.parametrize(
        "k,n,delta",
        [(1, 30, 0.05), (5, 50, 0.10), (12, 100, 0.20), (3, 2500, 0.001), (2, 20, 0.05)],
    )
    def test_the_bisected_branch_solves_the_defining_equation(self, k, n, delta):
        """A CP upper limit U is defined by `P(X <= k | n, U) = delta`. The
        bisection must land on it, not merely nearby."""
        upper = clopper_pearson_upper(k, n, delta)

        assert math.exp(_log_binomial_cdf(k, n, upper)) == pytest.approx(delta, rel=1e-9)

    @pytest.mark.parametrize("n,delta", [(10, 0.05), (299, 0.05), (2500, 0.001)])
    def test_the_zero_failure_shortcut_solves_the_same_equation(self, n, delta):
        """`k = 0` takes a closed form rather than the bisection; it must agree
        with the definition the bisected branch is held to."""
        upper = clopper_pearson_upper(0, n, delta)

        assert upper == pytest.approx(-math.expm1(math.log(delta) / n))
        assert math.exp(_log_binomial_cdf(0, n, upper)) == pytest.approx(delta, rel=1e-9)

    def test_an_all_failure_sample_yields_the_trivial_bound(self):
        assert clopper_pearson_upper(10, 10, 0.05) == 1.0

    @pytest.mark.parametrize(
        "k,n,delta",
        [(0, 10, 0.05), (1, 30, 0.05), (2, 20, 0.05), (5, 50, 0.1),
         (0, 2500, 0.001), (12, 100, 0.2), (99, 100, 0.05)],
    )
    def test_it_agrees_with_the_other_clopper_pearson_implementation_in_the_repo(
        self, k, n, delta
    ):
        """`scenario_labeling.cp_upper_bound` inverts a beta quantile; this one
        bisects a log-domain binomial CDF. They are used on the same corpus and
        must not drift apart."""
        assert clopper_pearson_upper(k, n, delta) == pytest.approx(
            cp_upper_bound_via_beta_quantile(k, n, delta), rel=1e-12
        )

    def test_the_bound_rises_with_observed_failures(self):
        bounds = [clopper_pearson_upper(k, 40, 0.05) for k in range(6)]

        assert bounds == sorted(bounds)

    def test_the_bound_tightens_as_evidence_accumulates(self):
        bounds = [clopper_pearson_upper(1, n, 0.05) for n in (10, 20, 50, 200)]

        assert bounds == sorted(bounds, reverse=True)

    @pytest.mark.parametrize("k,n", [(0, 20), (3, 40), (7, 100)])
    def test_the_bound_never_falls_below_the_point_estimate(self, k, n):
        assert clopper_pearson_upper(k, n, 0.05) >= k / n


# --------------------------------------------------------------------------
# Log-scale delta interval
# --------------------------------------------------------------------------


class TestLogScaleDeltaInterval:

    @pytest.mark.parametrize("rse", [None, float("nan"), math.inf, -math.inf])
    def test_an_unusable_standard_error_yields_no_interval(self, rse):
        assert _log_scale_delta_interval(math.log(0.1), rse, 0.05) == (None, None)

    def test_the_interval_is_multiplicative_around_the_estimate(self):
        """Built symmetrically in log space, so the endpoints are the estimate
        times and divided by the same factor -- not the estimate plus and minus
        the same amount."""
        estimate = 0.1
        lower, upper = _log_scale_delta_interval(math.log(estimate), 0.2, 0.05)

        assert upper / estimate == pytest.approx(estimate / lower)
        assert lower < estimate < upper

    def test_it_brackets_the_estimate(self):
        for rse in (0.01, 0.1, 0.5, 1.0):
            lower, upper = _log_scale_delta_interval(math.log(0.01), rse, 0.05)
            assert lower <= 0.01 <= upper

    def test_a_wider_standard_error_gives_a_wider_interval(self):
        widths = []
        for rse in (0.05, 0.1, 0.2, 0.4):
            lower, upper = _log_scale_delta_interval(math.log(0.01), rse, 0.05)
            widths.append(upper - lower)

        assert widths == sorted(widths)

    def test_a_higher_confidence_level_gives_a_wider_interval(self):
        widths = []
        for delta in (0.2, 0.1, 0.05, 0.01):
            lower, upper = _log_scale_delta_interval(math.log(0.01), 0.2, delta)
            widths.append(upper - lower)

        assert widths == sorted(widths)

    def test_a_zero_standard_error_collapses_the_interval_onto_the_estimate(self):
        lower, upper = _log_scale_delta_interval(math.log(0.1), 0.0, 0.05)

        assert lower == upper == pytest.approx(0.1)

    def test_a_negative_standard_error_is_clamped_rather_than_inverted(self):
        assert _log_scale_delta_interval(math.log(0.1), -5.0, 0.05) == (
            _log_scale_delta_interval(math.log(0.1), 0.0, 0.05)
        )

    def test_the_upper_endpoint_is_capped_at_certainty(self):
        """The quantity being bounded is a probability."""
        _, upper = _log_scale_delta_interval(math.log(0.9), 2.0, 0.05)

        assert upper == 1.0

    def test_an_unrepresentable_estimate_yields_no_endpoints(self):
        assert _log_scale_delta_interval(1000.0, 0.1, 0.05) == (None, None)

    def test_the_public_estimator_reports_this_interval_method(self):
        result = importance_sampling_estimate([0.0] * 100, [True] * 5 + [False] * 95)

        assert result["interval_method"] == "asymptotic-log-scale-normal-delta-two-sided"
        assert result["ci_lower"], result["ci_upper"] == _log_scale_delta_interval(
            result["log_estimate"], result["relative_standard_error"], result["delta"]
        )


# --------------------------------------------------------------------------
# Nominal Monte Carlo baseline
# --------------------------------------------------------------------------


class TestNaiveMonteCarloSummary:
    """The denominator of every reported variance-reduction factor."""

    def test_an_empty_sample_is_refused(self):
        with pytest.raises(ValueError, match="at least one sample"):
            naive_monte_carlo_summary([], delta=0.05)

    @pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
    def test_a_degenerate_confidence_level_is_refused(self, delta):
        with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
            naive_monte_carlo_summary([True, False], delta=delta)

    def test_it_reports_the_plain_failure_fraction(self):
        summary = naive_monte_carlo_summary([True, False, False, True], delta=0.05)

        assert summary["n"] == 4
        assert summary["failures"] == 2
        assert summary["estimate"] == 0.5

    def test_it_names_its_own_interval_method(self):
        summary = naive_monte_carlo_summary([False] * 10, delta=0.05)

        assert summary["interval_method"] == "exact-clopper-pearson-one-sided-upper"
        assert summary["delta"] == 0.05
        assert summary["confidence_level"] == 0.95

    def test_the_upper_bound_is_the_exact_clopper_pearson_limit(self):
        summary = naive_monte_carlo_summary([True] * 3 + [False] * 97, delta=0.05)

        assert summary["cp_upper"] == clopper_pearson_upper(3, 100, 0.05)

    @pytest.mark.parametrize("n", [1, 5, 20, 100])
    def test_the_point_estimate_never_exceeds_its_own_upper_bound(self, n):
        for k in range(n + 1):
            summary = naive_monte_carlo_summary([True] * k + [False] * (n - k), delta=0.05)
            assert summary["estimate"] <= summary["cp_upper"]

    def test_a_clean_run_still_reports_a_positive_upper_bound(self):
        """Zero observed failures is not evidence of zero risk; the bound is
        what carries the remaining uncertainty."""
        summary = naive_monte_carlo_summary([False] * 100, delta=0.05)

        assert summary["estimate"] == 0.0
        assert 0.0 < summary["cp_upper"] < 1.0

    def test_indicators_are_read_for_truthiness(self):
        summary = naive_monte_carlo_summary([1, 0, "", None, "x"], delta=0.05)

        assert (summary["n"], summary["failures"]) == (5, 2)

    def test_it_agrees_with_the_zero_failure_budget_helper(self):
        """`required_zero_failure_budget` says how many clean episodes are needed
        to bound a rate; the summary's bound at that count must actually clear
        it, and one episode short must not."""
        budget = required_zero_failure_budget(0.01, 0.05)

        assert naive_monte_carlo_summary([False] * budget, delta=0.05)["cp_upper"] <= 0.01
        assert naive_monte_carlo_summary([False] * (budget - 1), delta=0.05)["cp_upper"] > 0.01

    def test_the_baseline_bound_tightens_only_as_the_square_root_of_effort(self):
        """The scaling that makes rare-event estimation by nominal sampling
        impractical, and the reason the IS path exists at all."""
        hundred = naive_monte_carlo_summary([False] * 100, delta=0.05)["cp_upper"]
        ten_thousand = naive_monte_carlo_summary([False] * 10_000, delta=0.05)["cp_upper"]

        assert ten_thousand == pytest.approx(hundred / 100.0, rel=0.02)

    def test_importance_sampling_reaches_a_rate_the_nominal_baseline_cannot_see(self):
        """A same-budget comparison against a known Gaussian tail.

        On 20k episodes the nominal estimator never observes the event, so its
        point estimate is zero and all it can offer is a bound roughly five
        times the true rate. The weighted estimator recovers the rate to within
        a few percent and brackets it.
        """
        rng = random.Random(20260826)
        episodes = 20_000
        log_weights, biased_hits, nominal_hits = [], [], []
        for _ in range(episodes):
            biased = rng.gauss(4.0, 1.0)
            log_weights.append(0.5 * (biased - 4.0) ** 2 - 0.5 * biased ** 2)
            biased_hits.append(biased >= 4.0)
            nominal_hits.append(rng.gauss(0.0, 1.0) >= 4.0)

        truth = NormalDist().cdf(-4.0)
        weighted = importance_sampling_estimate(log_weights, biased_hits, delta=0.05)
        nominal = naive_monte_carlo_summary(nominal_hits, delta=0.05)

        assert nominal["failures"] == 0
        assert nominal["estimate"] == 0.0
        assert nominal["cp_upper"] > 4.0 * truth

        assert weighted["estimate"] == pytest.approx(truth, rel=0.05)
        assert weighted["ci_lower"] <= truth <= weighted["ci_upper"]
        assert weighted["ci_upper"] < nominal["cp_upper"]


# --------------------------------------------------------------------------
# Severity curve
# --------------------------------------------------------------------------


class TestSeverityCurve:

    DISTANCES = [0.1, 0.4, 0.6, 1.2, 2.0]
    NOMINAL_WEIGHTS = [0.0] * 5

    def test_it_returns_one_row_per_threshold_in_order(self):
        thresholds = [0.0, 0.5, 1.0, 1.5, 3.0]

        rows = severity_curve(
            self.DISTANCES, self.NOMINAL_WEIGHTS, thresholds, delta=0.05
        )

        assert [row["threshold"] for row in rows] == thresholds

    def test_no_thresholds_means_no_rows(self):
        assert severity_curve(self.DISTANCES, self.NOMINAL_WEIGHTS, [], delta=0.05) == []

    def test_each_row_carries_a_full_estimate_plus_its_threshold(self):
        rows = severity_curve(self.DISTANCES, self.NOMINAL_WEIGHTS, [1.0], delta=0.05)
        estimate = importance_sampling_estimate(
            self.NOMINAL_WEIGHTS, [True] * 5, delta=0.05
        )

        assert set(rows[0]) == set(estimate) | {"threshold"}

    def test_the_threshold_is_inclusive(self):
        """A distance exactly at the threshold counts as a violation."""
        rows = severity_curve([0.5], [0.0], [0.5], delta=0.05)

        assert rows[0]["failures_under_proposal"] == 1

    def test_the_curve_is_non_decreasing(self):
        rows = severity_curve(
            self.DISTANCES, self.NOMINAL_WEIGHTS, [0.0, 0.5, 1.0, 1.5, 3.0], delta=0.05
        )
        estimates = [row["estimate"] for row in rows]

        assert estimates == sorted(estimates)

    def test_a_threshold_below_every_observation_yields_no_events(self):
        rows = severity_curve(self.DISTANCES, self.NOMINAL_WEIGHTS, [0.0], delta=0.05)

        assert rows[0]["failures_under_proposal"] == 0
        assert rows[0]["estimate"] == 0.0
        assert rows[0]["ci_upper"] is None, "a zero-hit sample bounds nothing above"

    def test_a_threshold_above_every_observation_captures_all_of_them(self):
        rows = severity_curve(self.DISTANCES, self.NOMINAL_WEIGHTS, [10.0], delta=0.05)

        assert rows[0]["failures_under_proposal"] == 5
        assert rows[0]["estimate"] == pytest.approx(1.0)

    def test_the_confidence_level_reaches_every_row(self):
        rows = severity_curve(
            self.DISTANCES, self.NOMINAL_WEIGHTS, [0.5, 1.0], delta=0.01
        )

        assert all(row["delta"] == 0.01 for row in rows)
        assert all(row["confidence_level"] == 0.99 for row in rows)

    def test_one_weighted_sample_is_reused_across_the_whole_curve(self):
        """Every threshold re-scores the same episodes, so the overlap
        diagnostics are a property of the sample, not of the threshold."""
        rng = random.Random(7)
        distances, log_weights = [], []
        for _ in range(400):
            value = rng.gauss(3.0, 1.0)
            distances.append(value)
            log_weights.append(0.5 * (value - 3.0) ** 2 - 0.5 * value ** 2)

        rows = severity_curve(distances, log_weights, [0.0, 1.0, 2.0, 3.0, 4.0], delta=0.05)

        assert len({row["ess"] for row in rows}) == 1
        assert len({row["log_mean_weight"] for row in rows}) == 1
        assert all(row["n"] == 400 for row in rows)

    def test_the_curve_stays_monotone_under_non_uniform_weights(self):
        rng = random.Random(7)
        distances, log_weights = [], []
        for _ in range(400):
            value = rng.gauss(3.0, 1.0)
            distances.append(value)
            log_weights.append(0.5 * (value - 3.0) ** 2 - 0.5 * value ** 2)

        rows = severity_curve(
            distances, log_weights, [0.0, 1.0, 2.0, 3.0, 4.0, 10.0], delta=0.05
        )
        estimates = [row["estimate"] for row in rows]

        assert estimates == sorted(estimates)
        assert all(
            row["ci_lower"] is None or row["ci_lower"] <= row["estimate"] for row in rows
        )
        assert all(
            row["ci_upper"] is None or row["estimate"] <= row["ci_upper"] for row in rows
        )

    def test_malformed_weights_are_refused_before_any_row_is_produced(self):
        with pytest.raises(ValueError, match="non-empty and have equal length"):
            severity_curve([0.1, 0.2], [0.0], [1.0], delta=0.05)
