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

import math
import random
from statistics import NormalDist

import pytest

from source.design_exploration.rare_events.estimators import (
    assess_final_estimate,
    assess_proposal_validation,
    assess_weight_quality,
    clopper_pearson_upper,
    importance_sampling_estimate,
    required_zero_failure_budget,
    weight_diagnostics,
)


def test_nominal_weights_reduce_to_sample_mean_and_full_ess():
    result = importance_sampling_estimate([0.0] * 4, [False, True, False, True], delta=0.05)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ess"] == pytest.approx(4.0)
    assert result["snis_estimate"] == pytest.approx(0.5)


def test_extreme_log_weights_remain_stable():
    result = importance_sampling_estimate([-1000.0, -1001.0, -math.inf], [True, False, True])
    assert result["log_estimate"] == pytest.approx(-1000.0 - math.log(3.0))
    assert result["estimate"] == 0.0
    assert 1.0 <= result["ess"] <= 2.0


def test_zero_hit_importance_sample_does_not_claim_zero_width_upper_bound():
    result = importance_sampling_estimate([0.0] * 20, [False] * 20)
    assert result["estimate"] == 0.0
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] is None


def test_degenerate_weights_fail_explicit_quality_gates():
    diagnostics = weight_diagnostics([0.0] + [-60.0] * 9)
    thresholds = {
        "min_ess_ratio": 0.2,
        "max_normalized_weight": 0.5,
        "max_abs_log_mean_weight": 1.0,
        "min_proposal_failures": 2,
        "min_failure_ess": 2.0,
        "max_relative_standard_error": 0.5,
    }
    overlap = assess_weight_quality(diagnostics, thresholds)
    estimate = importance_sampling_estimate([0.0] + [-60.0] * 9, [False] * 9 + [True])
    final = assess_final_estimate(estimate, overlap, thresholds)
    assert not overlap["passed"]
    assert not final["passed"]
    assert "ess_ratio_below_minimum" in final["reasons"]
    assert "proposal_failures_below_minimum" in final["reasons"]


def test_ten_hit_estimate_is_asymmetric_but_fails_paper_precision_gates():
    log_weights = [0.0] * 2500
    estimate = importance_sampling_estimate(
        log_weights, [True] * 10 + [False] * 2490, delta=0.001
    )
    thresholds = {
        "min_ess_ratio": 0.25,
        "max_normalized_weight": 0.0025,
        "max_abs_log_mean_weight": 0.3,
        "min_proposal_failures": 50,
        "min_failure_ess": 50.0,
        "max_relative_standard_error": 0.12,
    }
    quality = assess_weight_quality(weight_diagnostics(log_weights), thresholds)
    assessment = assess_final_estimate(estimate, quality, thresholds)

    assert estimate["interval_method"] == (
        "asymptotic-log-scale-normal-delta-two-sided"
    )
    assert 0.0 < estimate["ci_lower"] < estimate["estimate"] < estimate["ci_upper"]
    assert estimate["failure_ess"] == pytest.approx(10.0)
    assert not assessment["passed"]
    assert "proposal_failures_below_minimum" in assessment["reasons"]
    assert "failure_ess_below_minimum" in assessment["reasons"]
    assert "relative_standard_error_above_maximum" in assessment["reasons"]


def test_final_gate_rejects_zero_lower_endpoint_even_if_other_gates_pass():
    estimate = importance_sampling_estimate(
        [0.0] * 2500, [True] * 100 + [False] * 2400, delta=0.001
    )
    estimate["ci_lower"] = 0.0
    thresholds = {
        "min_ess_ratio": 0.25,
        "max_normalized_weight": 0.0025,
        "max_abs_log_mean_weight": 0.3,
        "min_proposal_failures": 50,
        "min_failure_ess": 50.0,
        "max_relative_standard_error": 0.12,
    }
    quality = assess_weight_quality(weight_diagnostics([0.0] * 2500), thresholds)
    assessment = assess_final_estimate(estimate, quality, thresholds)

    assert not assessment["passed"]
    assert "positive_confidence_interval_lower_bound_unavailable" in assessment["reasons"]


def test_final_gate_rejects_degenerate_zero_width_interval():
    log_weights = [0.0] * 100
    estimate = importance_sampling_estimate(log_weights, [True] * 100, delta=0.001)
    thresholds = {
        "min_ess_ratio": 0.25,
        "max_normalized_weight": 0.02,
        "max_abs_log_mean_weight": 0.3,
        "min_proposal_failures": 75,
        "min_failure_ess": 50.0,
        "max_relative_standard_error": 0.12,
    }
    quality = assess_weight_quality(weight_diagnostics(log_weights), thresholds)
    assessment = assess_final_estimate(estimate, quality, thresholds)

    assert estimate["relative_standard_error"] == 0.0
    assert estimate["ci_lower"] == estimate["ci_upper"]
    assert not assessment["passed"]
    assert "positive_relative_standard_error_unavailable" in assessment["reasons"]


def test_defensive_weight_cap_does_not_make_tight_final_overlap_gates_vacuous():
    thresholds = {
        "min_ess_ratio": 0.25,
        "max_normalized_weight": 0.0025,
        "max_abs_log_mean_weight": 0.3,
    }
    low_ess_weights = [5.0] * 417 + [0.2] * 2083
    low_ess = assess_weight_quality(
        weight_diagnostics([math.log(value) for value in low_ess_weights]), thresholds
    )
    assert "ess_ratio_below_minimum" in low_ess["reasons"]

    high_leverage_weights = [5.0] + [0.75] * 2499
    high_leverage = assess_weight_quality(
        weight_diagnostics([math.log(value) for value in high_leverage_weights]), thresholds
    )
    assert "max_normalized_weight_above_maximum" in high_leverage["reasons"]


def test_proposal_validation_requires_events_and_projects_final_precision():
    thresholds = {
        "min_ess_ratio": 0.25,
        "max_normalized_weight": 0.08,
        "max_abs_log_mean_weight": 0.7,
        "min_proposal_failures": 2,
        "max_projected_final_relative_standard_error": 0.12,
    }
    log_weights = [0.0] * 50
    weight_quality = assess_weight_quality(weight_diagnostics(log_weights), thresholds)
    passing_estimate = importance_sampling_estimate(
        log_weights, [True, True, True] + [False] * 47
    )
    passing = assess_proposal_validation(
        passing_estimate, weight_quality, thresholds, final_episodes=2500
    )
    assert passing["passed"] is True
    assert passing["projected_final_relative_standard_error"] < 0.12

    uneven_logs = [math.log(2.0), math.log(0.01)] + [0.0] * 48
    uneven_quality = assess_weight_quality(weight_diagnostics(uneven_logs), thresholds)
    uneven_estimate = importance_sampling_estimate(
        uneven_logs, [True, True] + [False] * 48
    )
    imprecise = assess_proposal_validation(
        uneven_estimate, uneven_quality, thresholds, final_episodes=2500
    )
    assert uneven_quality["passed"] is True
    assert "projected_final_relative_standard_error_above_maximum" in imprecise["reasons"]

    zero_event_estimate = importance_sampling_estimate(log_weights, [False] * 50)
    rejected = assess_proposal_validation(
        zero_event_estimate, weight_quality, thresholds, final_episodes=2500
    )
    assert rejected["passed"] is False
    assert "proposal_validation_failures_below_minimum" in rejected["reasons"]
    assert "projected_final_relative_standard_error_above_maximum" in rejected["reasons"]


def test_paper_pointwise_clopper_pearson_budget_and_bound():
    bound = clopper_pearson_upper(0, 2500, 0.001)
    assert bound == pytest.approx(0.00275928825845, rel=1e-10)
    assert required_zero_failure_budget(1e-6, 0.001) == 6_907_752


def test_toy_gaussian_tail_importance_estimate():
    rng = random.Random(202503)
    proposal_mean = 4.0
    log_weights = []
    indicators = []
    for _ in range(50_000):
        value = rng.gauss(proposal_mean, 1.0)
        log_weights.append(0.5 * (value - proposal_mean) ** 2 - 0.5 * value ** 2)
        indicators.append(value >= 4.0)
    result = importance_sampling_estimate(log_weights, indicators)
    truth = NormalDist().cdf(-4.0)
    assert result["estimate"] == pytest.approx(truth, rel=0.04)
    assert result["relative_standard_error"] < 0.02
