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

"""Numerically stable importance-sampling and nominal-MC estimators."""

from __future__ import annotations

import math
import sys
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_LOG_MAX_FLOAT = math.log(sys.float_info.max)
_LOG_MIN_FLOAT = math.log(sys.float_info.min)
IS_INTERVAL_METHOD = "asymptotic-log-scale-normal-delta-two-sided"


def _representable_exp(log_value: float) -> Optional[float]:
    if log_value == -math.inf:
        return 0.0
    if not math.isfinite(log_value) or log_value > _LOG_MAX_FLOAT:
        return None
    if log_value < _LOG_MIN_FLOAT:
        return 0.0
    return math.exp(log_value)


def _validate_inputs(log_weights: Sequence[float], indicators: Sequence[bool]) -> None:
    if len(log_weights) != len(indicators) or not log_weights:
        raise ValueError("log_weights and indicators must be non-empty and have equal length.")
    if any(math.isnan(float(value)) or float(value) == math.inf for value in log_weights):
        raise ValueError("Log weights may be finite or -inf, but not NaN or +inf.")


def weight_diagnostics(log_weights: Sequence[float]) -> Dict[str, Optional[float]]:
    """Summarize proposal overlap without exponentiating extreme weights."""
    if not log_weights:
        raise ValueError("log_weights must be non-empty.")
    logs = [float(value) for value in log_weights]
    if any(math.isnan(value) or value == math.inf for value in logs):
        raise ValueError("Log weights may be finite or -inf, but not NaN or +inf.")
    finite_logs = [value for value in logs if value != -math.inf]
    n = len(logs)
    if not finite_logs:
        return {
            "n": n, "ess": 0.0, "ess_ratio": 0.0,
            "max_normalized_weight": None, "log_mean_weight": -math.inf,
            "mean_weight": 0.0,
        }
    shift = max(finite_logs)
    scaled = [0.0 if value == -math.inf else math.exp(value - shift) for value in logs]
    sum_w = sum(scaled)
    sum_w2 = sum(value * value for value in scaled)
    ess = 0.0 if sum_w2 == 0.0 else sum_w * sum_w / sum_w2
    log_mean = shift + math.log(sum_w / n)
    return {
        "n": n,
        "ess": ess,
        "ess_ratio": ess / n,
        "max_normalized_weight": None if sum_w == 0.0 else max(scaled) / sum_w,
        "log_mean_weight": log_mean,
        "mean_weight": _representable_exp(log_mean),
    }


def assess_weight_quality(
    diagnostics: Mapping[str, Optional[float]],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply configured overlap gates and return machine-readable reasons."""
    reasons = []
    ess_ratio = diagnostics.get("ess_ratio")
    max_weight = diagnostics.get("max_normalized_weight")
    log_mean = diagnostics.get("log_mean_weight")
    min_ess_ratio = float(thresholds["min_ess_ratio"])
    max_normalized = float(thresholds["max_normalized_weight"])
    max_abs_log_mean = float(thresholds["max_abs_log_mean_weight"])
    if ess_ratio is None or float(ess_ratio) < min_ess_ratio:
        reasons.append("ess_ratio_below_minimum")
    if max_weight is None or float(max_weight) > max_normalized:
        reasons.append("max_normalized_weight_above_maximum")
    if log_mean is None or not math.isfinite(float(log_mean)) or abs(float(log_mean)) > max_abs_log_mean:
        reasons.append("mean_weight_inconsistent_with_one")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            "min_ess_ratio": min_ess_ratio,
            "max_normalized_weight": max_normalized,
            "max_abs_log_mean_weight": max_abs_log_mean,
        },
    }


def assess_proposal_validation(
    estimate: Mapping[str, Optional[float]],
    weight_quality: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    final_episodes: int,
) -> Dict[str, Any]:
    """Decide whether a proposal merits the independent final-IS budget.

    The projected RSE uses the usual fixed-proposal ``1/sqrt(n)`` scaling.
    It is a pilot planning diagnostic, not a confidence interval and not part
    of the paper-facing estimator.
    """
    pilot_episodes = int(estimate.get("n") or 0)
    if pilot_episodes < 1 or int(final_episodes) < 1:
        raise ValueError("Pilot and final episode counts must be positive.")
    reasons = list(weight_quality.get("reasons", []))
    failures = int(estimate.get("failures_under_proposal") or 0)
    minimum_failures = int(thresholds["min_proposal_failures"])
    relative_se = estimate.get("relative_standard_error")
    projected_relative_se = (
        None
        if relative_se is None or not math.isfinite(float(relative_se))
        else float(relative_se) * math.sqrt(pilot_episodes / int(final_episodes))
    )
    maximum_projected_relative_se = float(
        thresholds["max_projected_final_relative_standard_error"]
    )
    if failures < minimum_failures:
        reasons.append("proposal_validation_failures_below_minimum")
    if (
        projected_relative_se is None
        or projected_relative_se > maximum_projected_relative_se
    ):
        reasons.append("projected_final_relative_standard_error_above_maximum")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "projected_final_relative_standard_error": projected_relative_se,
        "thresholds": {
            **dict(weight_quality.get("thresholds", {})),
            "min_proposal_failures": minimum_failures,
            "max_projected_final_relative_standard_error": maximum_projected_relative_se,
        },
    }


def assess_final_estimate(
    estimate: Mapping[str, Optional[float]],
    weight_quality: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Gate paper-facing output on overlap, event coverage, and precision."""
    reasons = list(weight_quality.get("reasons", []))
    failures = int(estimate.get("failures_under_proposal") or 0)
    min_failures = int(thresholds["min_proposal_failures"])
    failure_ess = estimate.get("failure_ess")
    min_failure_ess = float(thresholds["min_failure_ess"])
    relative_se = estimate.get("relative_standard_error")
    max_relative_se = float(thresholds["max_relative_standard_error"])
    if failures < min_failures:
        reasons.append("proposal_failures_below_minimum")
    if failure_ess is None or float(failure_ess) < min_failure_ess:
        reasons.append("failure_ess_below_minimum")
    if relative_se is None or not math.isfinite(float(relative_se)) or float(relative_se) > max_relative_se:
        reasons.append("relative_standard_error_above_maximum")
    elif float(relative_se) <= 0.0:
        reasons.append("positive_relative_standard_error_unavailable")
    ci_lower = estimate.get("ci_lower")
    ci_upper = estimate.get("ci_upper")
    if ci_lower is None or not math.isfinite(float(ci_lower)) or float(ci_lower) <= 0.0:
        reasons.append("positive_confidence_interval_lower_bound_unavailable")
    if ci_upper is None or not math.isfinite(float(ci_upper)):
        reasons.append("finite_confidence_interval_unavailable")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            **dict(weight_quality.get("thresholds", {})),
            "min_proposal_failures": min_failures,
            "min_failure_ess": min_failure_ess,
            "max_relative_standard_error": max_relative_se,
        },
    }


def _log_scale_delta_interval(
    log_estimate: float, relative_standard_error: Optional[float], delta: float,
) -> tuple[Optional[float], Optional[float]]:
    """Asymmetric delta-method interval for a strictly positive IS mean."""
    if relative_standard_error is None or not math.isfinite(relative_standard_error):
        return None, None
    z = NormalDist().inv_cdf(1.0 - delta / 2.0)
    half_width = z * max(float(relative_standard_error), 0.0)
    lower = _representable_exp(log_estimate - half_width)
    upper = _representable_exp(log_estimate + half_width)
    if upper is not None:
        upper = min(1.0, upper)
    return lower, upper


def importance_sampling_estimate(
    log_weights: Sequence[float],
    indicators: Sequence[bool],
    *,
    delta: float = 0.05,
) -> Dict[str, Any]:
    """Estimate ``E_q[I(E) p/q]`` and weight-degeneracy diagnostics."""
    _validate_inputs(log_weights, indicators)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    logs = [float(value) for value in log_weights]
    n = len(logs)
    overlap = weight_diagnostics(logs)
    finite_logs = [value for value in logs if value != -math.inf]
    if not finite_logs:
        return {
            "n": n,
            "failures_under_proposal": int(sum(bool(value) for value in indicators)),
            "estimate": 0.0,
            "log_estimate": -math.inf,
            "standard_error": 0.0,
            "relative_standard_error": None,
            "failure_ess": 0.0,
            "ci_lower": 0.0,
            "ci_upper": None,
            "interval_method": IS_INTERVAL_METHOD,
            "confidence_level": 1.0 - float(delta),
            "delta": float(delta),
            "snis_estimate": None,
            "ess": overlap["ess"],
            "ess_ratio": overlap["ess_ratio"],
            "max_normalized_weight": overlap["max_normalized_weight"],
            "log_mean_weight": overlap["log_mean_weight"],
            "mean_weight": overlap["mean_weight"],
        }

    shift = max(finite_logs)
    scaled_weights = [0.0 if value == -math.inf else math.exp(value - shift) for value in logs]
    scaled_values = [weight if bool(indicator) else 0.0 for weight, indicator in zip(scaled_weights, indicators)]
    sum_w = sum(scaled_weights)
    sum_y = sum(scaled_values)
    sum_y2 = sum(value * value for value in scaled_values)
    failure_ess = 0.0 if sum_y2 == 0.0 else sum_y * sum_y / sum_y2
    mean_y = sum_y / n
    log_estimate = -math.inf if mean_y == 0.0 else shift + math.log(mean_y)
    estimate = _representable_exp(log_estimate)

    if n > 1:
        variance_scaled = sum((value - mean_y) ** 2 for value in scaled_values) / (n - 1)
        se_scaled = math.sqrt(max(variance_scaled, 0.0) / n)
    else:
        se_scaled = 0.0
    log_se = -math.inf if se_scaled == 0.0 else shift + math.log(se_scaled)
    standard_error = _representable_exp(log_se)
    relative_se = None if mean_y == 0.0 else se_scaled / mean_y
    snis = None if sum_w == 0.0 else sum_y / sum_w

    if sum_y == 0.0:
        # A zero-hit importance sample does not justify a zero-width interval.
        # Unlike nominal Bernoulli MC, no distribution-free C-P analogue is
        # available without further assumptions on the likelihood ratio.
        ci_lower, ci_upper = 0.0, None
    elif estimate is None or standard_error is None:
        ci_lower = ci_upper = None
    else:
        ci_lower, ci_upper = _log_scale_delta_interval(
            log_estimate, relative_se, delta
        )

    return {
        "n": n,
        "failures_under_proposal": int(sum(bool(value) for value in indicators)),
        "estimate": estimate,
        "log_estimate": log_estimate,
        "standard_error": standard_error,
        "relative_standard_error": relative_se,
        "failure_ess": failure_ess,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "interval_method": IS_INTERVAL_METHOD,
        "confidence_level": 1.0 - float(delta),
        "delta": float(delta),
        "snis_estimate": snis,
        "ess": overlap["ess"],
        "ess_ratio": overlap["ess_ratio"],
        "max_normalized_weight": overlap["max_normalized_weight"],
        "log_mean_weight": overlap["log_mean_weight"],
        "mean_weight": overlap["mean_weight"],
    }


def _log_binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if k >= n else -math.inf
    terms = []
    log_p = math.log(probability)
    log_1mp = math.log1p(-probability)
    for i in range(k + 1):
        terms.append(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_1mp
        )
    shift = max(terms)
    return shift + math.log(sum(math.exp(term - shift) for term in terms))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided exact binomial upper bound with failure probability ``delta``."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError("Require n >= 1 and 0 <= k <= n.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(delta) / n)
    target = math.log(delta)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _log_binomial_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def naive_monte_carlo_summary(indicators: Sequence[bool], *, delta: float) -> Dict[str, Any]:
    if not indicators:
        raise ValueError("Nominal MC requires at least one sample.")
    n = len(indicators)
    failures = int(sum(bool(value) for value in indicators))
    return {
        "n": n,
        "failures": failures,
        "estimate": failures / n,
        "cp_upper": clopper_pearson_upper(failures, n, delta),
        "delta": float(delta),
        "confidence_level": 1.0 - float(delta),
        "interval_method": "exact-clopper-pearson-one-sided-upper",
    }


def required_zero_failure_budget(target_probability: float, delta: float) -> int:
    if not 0.0 < target_probability < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("target_probability and delta must be in (0, 1).")
    return int(math.ceil(math.log(delta) / math.log1p(-target_probability)))


def severity_curve(
    min_distances: Sequence[float],
    log_weights: Sequence[float],
    thresholds: Iterable[float],
    *,
    delta: float,
) -> List[Dict[str, Optional[float]]]:
    rows = []
    for threshold in thresholds:
        estimate = importance_sampling_estimate(
            log_weights,
            [float(value) <= float(threshold) for value in min_distances],
            delta=delta,
        )
        rows.append({"threshold": float(threshold), **estimate})
    return rows
