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
"""
Campaign endpoint estimators for the rare-event study.

A campaign records one row per episode, carrying its likelihood ratio
(`log_weight`), the failure indicator, and the route features used to classify
it. This module turns those rows into the reported quantities:

* `estimate_mode` dispatches on the sampling scheme a job was run under --
  a single IID proposal, a deterministic mixture, or the nominal distribution --
  and returns the estimate with its interval and precision diagnostics.
* `estimate_family_vector` estimates the three route families jointly, with
  their covariance, so contrasts between families are paired.
* `classify_family` and `PRINCIPAL_MODE` define those route partitions.

The estimators read the stored row fields. Re-deriving those fields from the
simulator logs is the campaign's own audit and needs the trajectories.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta

from source.design_exploration.rare_events.estimators import (
    clopper_pearson_upper,
    importance_sampling_estimate,
    weight_diagnostics,
)

# The route signature singled out as the principal failure mode. It was
# selected after observing cross-design presence, so it is an exploratory
# endpoint, not a pre-registered one.
PRINCIPAL_MODE = (
    "det=early|out=before_detection|fresh=late|age=fresh"
    "|brake=normal|ghost=0|wrong=0|miss=1|drop=0"
)

# Exhaustive partition of failures by detection timing and first-delivery age.
FAMILIES = (
    "late_detection",
    "early_detection_stale_first_delivery",
    "early_detection_fresh_first_delivery",
)
FAMILY_LABELS = {
    "late_detection": "Late/no detection",
    "early_detection_stale_first_delivery": "Early detection + stale/no delivery",
    "early_detection_fresh_first_delivery": "Early detection + fresh first delivery",
}
DETECTION_CUT_MS = 2990
RECEIPT_AGE_CUT_MS = 200

# Predeclared gates below which a weighted estimate is reported as exploratory.
PRECISION_GATES = {
    "minimum_event_paths": 20,
    "minimum_event_ess": 20.0,
    "maximum_event_contribution": 0.20,
    "maximum_relative_standard_error": 0.25,
}
ALPHA = 0.05

def classify_family(
    row: Mapping[str, Any], *, detection_cut_ms: int = DETECTION_CUT_MS,
    receipt_age_cut_ms: int = RECEIPT_AGE_CUT_MS,
) -> Optional[str]:
    """Return the exhaustive route-family label for a frozen failure row."""
    if not bool(row.get("failure")):
        return None
    detection = row.get("first_detection_ms")
    age = row.get("first_external_message_age_ms")
    if detection is None or float(detection) > detection_cut_ms:
        return "late_detection"
    if age is None or float(age) > receipt_age_cut_ms:
        return "early_detection_stale_first_delivery"
    return "early_detection_fresh_first_delivery"

def normal_interval(estimate: float, rse: Optional[float], alpha: float) -> Tuple[Optional[float], Optional[float]]:
    if estimate <= 0.0 or rse is None or not math.isfinite(rse):
        return 0.0, None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return (
        math.exp(math.log(estimate) - z * rse),
        min(1.0, math.exp(math.log(estimate) + z * rse)),
    )

def clopper_pearson_interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper

def clopper_pearson_one_sided_upper(k: int, n: int, alpha: float) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha, k + 1, n - k))

def _max_contribution(values: Sequence[float]) -> Optional[float]:
    total = math.fsum(values)
    if total <= 0.0:
        return None
    return max(values) / total

def stratified_estimate(
    rows: Sequence[Mapping[str, Any]],
    log_weights: Optional[Sequence[float]] = None,
    *,
    indicator: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Estimate an event under a fixed, deterministic mixture allocation.

    The default output retains the aggregate-failure schema. With an explicit
    binary indicator, estimate/SE/CI describe that event and additional
    events_under_proposal/event_ess fields describe its count and ESS.
    failures_under_proposal/failure_ess always describe the original failures.
    Rows and likelihood weights are never recoded or mutated. This helper
    requires sampling_component; IID corpora need their own estimator path.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("Stratified estimation requires nonempty rows.")
    logs = [float(row["log_weight"]) for row in rows] if log_weights is None else list(log_weights)
    if len(logs) != n:
        raise ValueError("log_weights must have one value per row.")
    if any(not math.isfinite(value) and value != -math.inf for value in logs):
        raise ValueError("log_weights must be finite or negative infinity.")
    failures = [bool(row["failure"]) for row in rows]
    events = failures if indicator is None else list(indicator)
    if len(events) != n:
        raise ValueError("indicator must have one value per row.")
    if any(value not in (False, True) for value in events):
        raise ValueError("indicator values must be binary (0 or 1).")
    if any("sampling_component" not in row for row in rows):
        raise ValueError(
            "Stratified estimation requires sampling_component on every row; "
            "IID corpora need their own estimator path."
        )
    weights = [0.0 if value == -math.inf else math.exp(value) for value in logs]
    event_values = [weight if event else 0.0 for event, weight in zip(events, weights)]
    estimate = math.fsum(event_values) / n
    groups: Dict[str, List[float]] = {}
    for row, value in zip(rows, event_values):
        groups.setdefault(str(row["sampling_component"]), []).append(value)
    variance = 0.0
    for values in groups.values():
        if len(values) < 2:
            raise ValueError("Every deterministic-mixture stratum needs at least two samples.")
        mean = math.fsum(values) / len(values)
        sample_variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
        variance += len(values) * sample_variance / (n * n)
    standard_error = math.sqrt(max(variance, 0.0))
    rse = None if estimate <= 0.0 else standard_error / estimate
    z = NormalDist().inv_cdf(0.975)
    ci = (None, None) if rse is None else (
        math.exp(math.log(estimate) - z * rse),
        min(1.0, math.exp(math.log(estimate) + z * rse)),
    )
    event_sum = math.fsum(event_values)
    event_sum_sq = math.fsum(value * value for value in event_values)
    diagnostics = weight_diagnostics(logs)
    result = {
        **diagnostics,
        "n": n,
        "estimate": estimate,
        "standard_error": standard_error,
        "relative_standard_error": rse,
        "ci_lower": ci[0], "ci_upper": ci[1], "confidence_level": 0.95,
        "interval_method": "stratified-asymptotic-log-scale-normal-delta-two-sided",
        "failure_ess": 0.0 if event_sum_sq == 0.0 else event_sum * event_sum / event_sum_sq,
        "failures_under_proposal": sum(failures),
        "stratum_counts": {name: len(values) for name, values in sorted(groups.items())},
    }
    if indicator is not None:
        result["events_under_proposal"] = sum(bool(event) for event in events)
        result["event_ess"] = result["failure_ess"]
        failure_values = [weight if failure else 0.0 for failure, weight in zip(failures, weights)]
        failure_sum = math.fsum(failure_values)
        failure_sum_sq = math.fsum(value * value for value in failure_values)
        result["failure_ess"] = (
            0.0 if failure_sum_sq == 0.0 else failure_sum * failure_sum / failure_sum_sq
        )
    return result

def estimate_family_vector(
    rows: Sequence[Mapping[str, Any]], scheme: str, *,
    detection_cut_ms: int = DETECTION_CUT_MS,
    receipt_age_cut_ms: int = RECEIPT_AGE_CUT_MS,
) -> Dict[str, Any]:
    """Joint family estimator, including covariance of the three estimates."""
    n = len(rows)
    if n < 2:
        raise ValueError("At least two rows are required.")
    labels = [
        classify_family(
            row, detection_cut_ms=detection_cut_ms,
            receipt_age_cut_ms=receipt_age_cut_ms,
        )
        for row in rows
    ]
    weights = np.asarray([
        0.0 if float(row["log_weight"]) == -math.inf else math.exp(float(row["log_weight"]))
        for row in rows
    ], dtype=float)
    y = np.zeros((n, len(FAMILIES)), dtype=float)
    for column, family in enumerate(FAMILIES):
        y[:, column] = weights * np.asarray([label == family for label in labels], dtype=float)

    if scheme == "iid_is":
        estimates = y.mean(axis=0)
        covariance = np.cov(y, rowvar=False, ddof=1) / n
        stratum_counts = None
    elif scheme == "nominal_mc":
        if not np.allclose(weights, 1.0, rtol=0.0, atol=1e-15):
            raise ValueError("Nominal-MC rows must have unit likelihood weights.")
        estimates = y.mean(axis=0)
        covariance = (np.diag(estimates) - np.outer(estimates, estimates)) / n
        stratum_counts = None
    elif scheme == "stratified_is":
        if any("sampling_component" not in row for row in rows):
            raise ValueError("Stratified rows require sampling_component.")
        estimates = y.mean(axis=0)
        covariance = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=float)
        groups: Dict[str, List[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            groups[str(row["sampling_component"])].append(position)
        for positions in groups.values():
            if len(positions) < 2:
                raise ValueError("Every deterministic stratum needs at least two rows.")
            group_cov = np.cov(y[positions, :], rowvar=False, ddof=1)
            covariance += len(positions) * group_cov / (n * n)
        stratum_counts = dict(sorted((name, len(values)) for name, values in groups.items()))
    else:
        raise ValueError(f"Unknown estimator scheme {scheme!r}.")

    counts = Counter(label for label in labels if label is not None)
    if sum(counts.values()) != sum(bool(row["failure"]) for row in rows):
        raise ValueError("The route-family classifier did not partition every failure.")
    family_sum = float(estimates.sum())
    direct_total = float(np.mean(weights * np.asarray([bool(row["failure"]) for row in rows])))
    if not math.isclose(family_sum, direct_total, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Family estimates do not sum to the aggregate estimate.")

    simultaneous_alpha = ALPHA / len(FAMILIES)
    records: Dict[str, Any] = {}
    total_variance = float(np.ones(len(FAMILIES)) @ covariance @ np.ones(len(FAMILIES)))
    for column, family in enumerate(FAMILIES):
        estimate = float(estimates[column])
        se = math.sqrt(max(0.0, float(covariance[column, column])))
        rse = None if estimate <= 0.0 else se / estimate
        event_values = y[:, column]
        event_sum = float(event_values.sum())
        event_sum_sq = float(np.square(event_values).sum())
        ess = 0.0 if event_sum_sq == 0.0 else event_sum * event_sum / event_sum_sq
        max_contribution = None if event_sum <= 0.0 else float(event_values.max() / event_sum)
        if scheme == "nominal_mc":
            pointwise = clopper_pearson_interval(counts[family], n, ALPHA)
            simultaneous = clopper_pearson_interval(counts[family], n, simultaneous_alpha)
            one_sided = clopper_pearson_one_sided_upper(counts[family], n, ALPHA)
            interval_method = "exact-clopper-pearson"
        else:
            pointwise = normal_interval(estimate, rse, ALPHA)
            simultaneous = normal_interval(estimate, rse, simultaneous_alpha)
            one_sided = None
            interval_method = "asymptotic-log-delta"
        records[family] = {
            "events_under_proposal": counts[family],
            "estimate": estimate,
            "standard_error": se,
            "relative_standard_error": rse,
            "pointwise_ci95": {"lower": pointwise[0], "upper": pointwise[1]},
            "within_design_family_simultaneous_ci95": {
                "lower": simultaneous[0], "upper": simultaneous[1],
                "bonferroni_family_count": len(FAMILIES),
            },
            "one_sided_cp_upper95": one_sided,
            "event_ess": None if scheme == "nominal_mc" else ess,
            "maximum_event_contribution": None if scheme == "nominal_mc" else max_contribution,
            "interval_method": interval_method,
        }

    # Conditional share P(family | failure), useful descriptively but not a
    # replacement for the joint probabilities used for robustness ranking.
    total_gradient_base = np.ones(len(FAMILIES))
    for column, family in enumerate(FAMILIES):
        share = float(estimates[column] / family_sum) if family_sum > 0.0 else None
        share_se = None
        share_ci = {"lower": None, "upper": None}
        if share is not None and 0.0 < share < 1.0:
            gradient = np.zeros(len(FAMILIES))
            gradient[column] = 1.0 / family_sum
            gradient -= estimates[column] * total_gradient_base / (family_sum * family_sum)
            share_se = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
            logit = math.log(share / (1.0 - share))
            logit_se = share_se / (share * (1.0 - share))
            z = NormalDist().inv_cdf(0.975)
            low_logit, high_logit = logit - z * logit_se, logit + z * logit_se
            share_ci = {
                "lower": 1.0 / (1.0 + math.exp(-low_logit)),
                "upper": 1.0 / (1.0 + math.exp(-high_logit)),
            }
        records[family]["conditional_share_of_failure"] = {
            "estimate": share, "standard_error_delta": share_se,
            "pointwise_ci95_logit_delta": share_ci,
        }

    total_se = math.sqrt(max(0.0, total_variance))
    return {
        "n": n,
        "failures_under_proposal": sum(counts.values()),
        "family_counts": {family: counts[family] for family in FAMILIES},
        "family_estimates": records,
        "covariance_matrix": covariance.tolist(),
        "covariance_order": list(FAMILIES),
        "aggregate": {
            "estimate": family_sum,
            "standard_error": total_se,
            "relative_standard_error": None if family_sum <= 0.0 else total_se / family_sum,
            "partition_residual": family_sum - direct_total,
        },
        "stratum_counts": stratum_counts,
    }

def precision_status(record: Mapping[str, Any], scheme: str) -> Dict[str, Any]:
    count = int(record["events_under_proposal"])
    if count == 0:
        return {
            "status": "not_observed",
            "failed_gates": None,
            "reason": "Zero IS hits supply no distribution-free upper bound." if scheme != "nominal_mc" else
                      "Zero nominal-MC hits; use the exact one-sided upper bound.",
        }
    if scheme == "nominal_mc":
        rse = record["relative_standard_error"]
        return {
            "status": "mc_exact_but_imprecise" if count < 20 or (rse or math.inf) > 0.25 else "mc_exact_precise",
            "failed_gates": [
                name for name, ok in {
                    "minimum_event_paths": count >= 20,
                    "maximum_relative_standard_error": rse is not None and rse <= 0.25,
                }.items() if not ok
            ],
            "reason": "The interval construction is exact; this status describes width, not validity.",
        }
    checks = {
        "minimum_event_paths": count >= PRECISION_GATES["minimum_event_paths"],
        "minimum_event_ess": float(record["event_ess"] or 0.0) >= PRECISION_GATES["minimum_event_ess"],
        "maximum_event_contribution": float(record["maximum_event_contribution"] or 1.0) <= PRECISION_GATES["maximum_event_contribution"],
        "maximum_relative_standard_error": float(record["relative_standard_error"] or math.inf) <= PRECISION_GATES["maximum_relative_standard_error"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "precise" if not failed else "imprecise_exploratory",
        "gate_checks": checks,
        "failed_gates": failed,
        "reason": None if not failed else "Valid weighted estimate, but one or more uniform reporting gates fail.",
    }

def estimate_mode(
    design: str, spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], hits: Sequence[bool],
) -> Dict[str, Any]:
    """Dispatch on the audited sampling scheme, then validate its fields."""
    scheme = spec["scheme"]
    logs = [float(row["log_weight"]) for row in rows]
    weights = [0.0 if value == -math.inf else math.exp(value) for value in logs]
    hit_weights = [w for w, hit in zip(weights, hits) if hit]
    count = sum(1 for hit in hits if hit)

    if scheme == "deterministic_mixture":
        if any("sampling_component" not in row for row in rows):
            raise ValueError(f"{design}: mixture corpus lacks sampling_component.")
        if any("proposal_log_ratios" not in row for row in rows):
            raise ValueError(f"{design}: mixture corpus lacks proposal_log_ratios.")
        result = stratified_estimate(rows, indicator=list(hits))
        out = {
            "estimator": "stratified",
            "estimate": result["estimate"],
            "standard_error": result["standard_error"],
            "relative_standard_error": result["relative_standard_error"],
            "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
            "mode_events": result["events_under_proposal"],
            "mode_ess": result["event_ess"],
            "aggregate_failures": result["failures_under_proposal"],
            "aggregate_ess": result["failure_ess"],
            "stratum_counts": result["stratum_counts"],
            "interval_method": result["interval_method"],
        }
    elif scheme == "iid_single_proposal":
        if any("sampling_component" in row for row in rows):
            raise ValueError(f"{design}: mixture fields on an IID-registered corpus.")
        result = importance_sampling_estimate(logs, list(hits))
        out = {
            "estimator": "iid_weighted",
            "estimate": result["estimate"],
            "standard_error": result["standard_error"],
            "relative_standard_error": result["relative_standard_error"],
            "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
            "mode_events": result["failures_under_proposal"],
            "mode_ess": result["failure_ess"],
            "aggregate_failures": sum(1 for row in rows if row.get("failure")),
            "interval_method": result["interval_method"],
        }
    elif scheme == "nominal_mc":
        lower, upper = clopper_pearson_interval(count, len(rows), 0.05)
        out = {
            "estimator": "exact_binomial",
            "estimate": count / len(rows),
            "standard_error": math.sqrt(
                (count / len(rows)) * (1 - count / len(rows)) / len(rows)
            ),
            "relative_standard_error": None,
            "ci_lower": lower, "ci_upper": upper,
            "mode_events": count, "mode_ess": None,
            "aggregate_failures": sum(1 for row in rows if row.get("failure")),
            "interval_method": "exact-clopper-pearson-two-sided",
            "one_sided_upper": clopper_pearson_upper(count, len(rows), 0.05),
        }
    else:
        raise ValueError(f"{design}: unknown sampling scheme {scheme!r}.")

    out["max_event_contribution"] = (
        None if scheme == "nominal_mc" else _max_contribution(hit_weights)
    )
    return out

def mode_status(scheme: str, estimate: Mapping[str, Any]) -> Dict[str, Any]:
    """Mode-specific reporting status under the campaign's own precision gates."""
    if estimate["mode_events"] == 0:
        return {
            "status": "not_observed",
            "failed_gates": None,
            "reason": "Zero mode events; establishes neither zero nominal risk "
                      "nor a bound from the IS episode count.",
        }
    if scheme == "nominal_mc":
        return {
            "status": "mc_exact",
            "failed_gates": None,
            "reason": "Exact binomial inference; weighted ESS and contribution "
                      "gates do not apply to an unweighted nominal sample.",
        }
    checks = {
        "minimum_event_paths":
            estimate["mode_events"] >= PRECISION_GATES["minimum_event_paths"],
        "minimum_event_ess":
            (estimate["mode_ess"] or 0.0) >= PRECISION_GATES["minimum_event_ess"],
        "maximum_event_contribution":
            (estimate["max_event_contribution"] or 1.0)
            <= PRECISION_GATES["maximum_event_contribution"],
        "maximum_relative_standard_error":
            (estimate["relative_standard_error"] or 1.0)
            <= PRECISION_GATES["maximum_relative_standard_error"],
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    return {
        "status": "precise" if not failed else "imprecise_exploratory",
        "gate_checks": checks,
        "failed_gates": failed,
        "reason": None if not failed else
        "Mode estimate is valid but does not meet the campaign's predeclared "
        "precision gates; report as exploratory.",
    }
