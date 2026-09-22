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
"""Pin the primitives every SAFE/UNSAFE label in the corpus is computed from.

`test_v3_recovery.py` already checks that the online and offline labellers agree
with each other. That is a consistency check: it passes even if both paths share
the same wrong bound. These tests instead check each primitive against an
external definition --

  * `cp_upper_bound` against the binomial CDF identity that *defines* a
    Clopper-Pearson upper limit, plus a closed form at k=0 and an exact coverage
    computation;
  * `required_min_iters_k0` against its own inequality and against
    `cp_upper_bound`, which must agree on the same n;
  * `sprt_log_likelihood` against the Wald ratio written out term by term.

-- and then pin the stopping rules those primitives feed, including which label
modes the shipped configuration can actually produce.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import binom

from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.offline_labeling import offline_sprt_label
from source.design_exploration.commons.scenario_labeling import (
    CAP_UNSAFE,
    CP_CAP,
    CP_K0,
    OBS_FAIL,
    WALD_SAFE,
    WALD_UNSAFE,
    ScenarioTester,
    SPRTConfig,
    SPRTDecision,
    SPRTState,
    TestResult as ScenarioTestResult,
    cp_upper_bound,
    required_min_iters_k0,
    sprt_decision_for_counts,
    sprt_log_likelihood,
)


SHIPPED = SPRTConfig()
ALPHA = 1.0 - SHIPPED.C


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact binomial CDF in pure Python -- no scipy, no shared code path."""
    return sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k + 1))


# --------------------------------------------------------------------------
# Clopper-Pearson upper confidence limit
# --------------------------------------------------------------------------


class TestClopperPearsonDefinition:
    """A one-sided CP upper limit U at level delta is *defined* by
    `P(X <= k | n, U) = delta`. Checking that identity validates the bound
    independently of the beta-quantile route used to compute it."""

    @pytest.mark.parametrize(
        "k,n,delta",
        [
            (0, 10, 0.05),
            (1, 10, 0.05),
            (2, 20, 0.05),
            (5, 50, 0.10),
            (1, 80, 0.05),
            (7, 30, 0.01),
            (12, 100, 0.20),
        ],
    )
    def test_the_bound_solves_the_defining_binomial_equation(self, k, n, delta):
        upper = cp_upper_bound(k, n, delta)

        assert binom.cdf(k, n, upper) == pytest.approx(delta, abs=1e-9)

    @pytest.mark.parametrize("k,n,delta", [(0, 10, 0.05), (2, 20, 0.05), (3, 25, 0.10)])
    def test_the_identity_holds_against_a_scipy_free_reference(self, k, n, delta):
        upper = cp_upper_bound(k, n, delta)

        assert _binomial_cdf(k, n, upper) == pytest.approx(delta, abs=1e-9)

    @pytest.mark.parametrize("n,delta", [(1, 0.05), (10, 0.05), (59, 0.05), (80, 0.01)])
    def test_the_zero_failure_case_matches_its_closed_form(self, n, delta):
        """With k = 0 the equation collapses to `(1 - U)**n = delta`."""
        assert cp_upper_bound(0, n, delta) == pytest.approx(1.0 - delta ** (1.0 / n))

    @pytest.mark.parametrize(
        "n,delta,true_p",
        [(20, 0.05, 0.05), (30, 0.10, 0.20), (15, 0.05, 0.30), (25, 0.05, 0.10)],
    )
    def test_the_bound_attains_its_nominal_coverage(self, n, delta, true_p):
        """Exact (not simulated) coverage: sum the binomial mass over every
        outcome whose bound covers the true rate."""
        coverage = sum(
            binom.pmf(k, n, true_p)
            for k in range(n + 1)
            if cp_upper_bound(k, n, delta) >= true_p
        )

        assert coverage >= 1.0 - delta


class TestClopperPearsonBoundary:

    def test_no_trials_yields_no_information(self):
        assert cp_upper_bound(0, 0, 0.05) == 1.0

    @pytest.mark.parametrize("k,n", [(3, 3), (10, 10), (5, 3), (1, 1)])
    def test_an_all_failure_or_impossible_count_yields_the_trivial_bound(self, k, n):
        assert cp_upper_bound(k, n, 0.05) == 1.0

    @pytest.mark.parametrize("k,n,delta", [(0, 10, 0.05), (3, 40, 0.1), (0, 1, 0.5)])
    def test_the_bound_is_a_probability(self, k, n, delta):
        assert 0.0 <= cp_upper_bound(k, n, delta) <= 1.0

    def test_the_bound_rises_with_observed_failures(self):
        bounds = [cp_upper_bound(k, 20, ALPHA) for k in range(6)]

        assert bounds == sorted(bounds)
        assert len(set(bounds)) == len(bounds)

    def test_the_bound_tightens_as_evidence_accumulates(self):
        bounds = [cp_upper_bound(1, n, ALPHA) for n in (5, 10, 20, 40, 80)]

        assert bounds == sorted(bounds, reverse=True)

    def test_the_bound_tightens_as_confidence_is_relaxed(self):
        bounds = [cp_upper_bound(2, 20, d) for d in (0.01, 0.05, 0.10, 0.20)]

        assert bounds == sorted(bounds, reverse=True)

    @pytest.mark.parametrize("delta,expected", [(0.0, 1.0), (1.0, 0.0)])
    def test_degenerate_confidence_levels_collapse_the_bound(self, delta, expected):
        """Characterisation: delta is not range-checked, so these pass through to
        the beta quantile and return the degenerate limits."""
        assert cp_upper_bound(2, 10, delta) == expected


# --------------------------------------------------------------------------
# Zero-failure budget
# --------------------------------------------------------------------------


class TestRequiredMinItersK0:
    """`required_min_iters_k0(P, alpha)` is the smallest n for which observing no
    failure rules out a failure rate of P at level alpha."""

    @pytest.mark.parametrize(
        "P,alpha", [(0.05, 0.05), (0.10, 0.05), (0.01, 0.10), (0.05, 0.01), (0.2, 0.05)]
    )
    def test_n_is_sufficient_and_minimal(self, P, alpha):
        n = required_min_iters_k0(P, alpha)

        assert (1.0 - P) ** n <= alpha, "n must rule the failure rate out"
        assert (1.0 - P) ** (n - 1) > alpha, "n - 1 must not"

    @pytest.mark.parametrize(
        "P,alpha", [(0.05, 0.05), (0.10, 0.05), (0.01, 0.10), (0.05, 0.01), (0.2, 0.05)]
    )
    def test_it_agrees_with_the_clopper_pearson_bound_on_the_same_n(self, P, alpha):
        """The two functions are separate implementations of one threshold. If
        they ever disagree, a design labelled SAFE by the k=0 rule would not
        survive its own confidence bound."""
        n = required_min_iters_k0(P, alpha)

        assert cp_upper_bound(0, n, alpha) <= P
        assert cp_upper_bound(0, n - 1, alpha) > P

    def test_the_shipped_configuration_needs_fifty_nine_clean_episodes(self):
        assert required_min_iters_k0(SHIPPED.P, ALPHA) == 59

    def test_a_stricter_target_rate_costs_more_episodes(self):
        budgets = [required_min_iters_k0(P, 0.05) for P in (0.2, 0.1, 0.05, 0.01)]

        assert budgets == sorted(budgets)

    def test_a_stricter_confidence_level_costs_more_episodes(self):
        budgets = [required_min_iters_k0(0.05, a) for a in (0.2, 0.1, 0.05, 0.01)]

        assert budgets == sorted(budgets)

    def test_a_vacuous_confidence_level_demands_nothing(self):
        assert required_min_iters_k0(0.05, 1.0) == 0

    @pytest.mark.parametrize(
        "P,error", [(0.0, ZeroDivisionError), (1.0, ValueError)]
    )
    def test_degenerate_target_rates_are_unguarded(self, P, error):
        """Characterisation: the function does not validate P. Callers reach it
        through `SPRTConfig`, which does not validate P either."""
        with pytest.raises(error):
            required_min_iters_k0(P, 0.05)


# --------------------------------------------------------------------------
# Wald log-likelihood ratio
# --------------------------------------------------------------------------


class TestSprtLogLikelihood:

    @pytest.mark.parametrize("n,k", [(-1, 0), (0, -1), (2, 3), (5, 6), (-3, -3)])
    def test_impossible_counts_are_rejected(self, n, k):
        with pytest.raises(ValueError, match="Invalid SPRT counts"):
            sprt_log_likelihood(n, k, SHIPPED)

    def test_no_evidence_is_no_evidence(self):
        assert sprt_log_likelihood(0, 0, SHIPPED) == 0.0

    @pytest.mark.parametrize("n,k", [(1, 0), (10, 3), (80, 5), (59, 0), (7, 7)])
    def test_it_matches_the_wald_ratio_written_out(self, n, k):
        p0 = SHIPPED.P
        p1 = SHIPPED.p1_factor * SHIPPED.P
        expected = k * math.log(p1 / p0) + (n - k) * math.log((1.0 - p1) / (1.0 - p0))

        assert sprt_log_likelihood(n, k, SHIPPED) == pytest.approx(expected)

    def test_a_clean_episode_is_evidence_for_safety(self):
        step = sprt_log_likelihood(1, 0, SHIPPED) - sprt_log_likelihood(0, 0, SHIPPED)

        assert step > 0
        assert step == pytest.approx(math.log((1.0 - 0.025) / (1.0 - 0.05)))

    def test_a_failure_is_evidence_against_safety(self):
        step = sprt_log_likelihood(1, 1, SHIPPED) - sprt_log_likelihood(0, 0, SHIPPED)

        assert step < 0
        assert step == pytest.approx(math.log(0.025 / 0.05))

    def test_a_failure_outweighs_many_clean_episodes(self):
        """One failure costs ~26 clean episodes under the shipped config, which
        is why a single violation dominates the trajectory."""
        safe_step = sprt_log_likelihood(1, 0, SHIPPED)
        fail_step = -sprt_log_likelihood(1, 1, SHIPPED)

        assert fail_step / safe_step == pytest.approx(26.68, rel=1e-3)

    @pytest.mark.parametrize("n,k", [(10, 3), (50, 7), (80, 0)])
    def test_the_ratio_decomposes_into_its_two_terms(self, n, k):
        total = sprt_log_likelihood(n, k, SHIPPED)
        failures = sprt_log_likelihood(k, k, SHIPPED)
        successes = sprt_log_likelihood(n - k, 0, SHIPPED)

        assert total == pytest.approx(failures + successes)

    def test_it_depends_only_on_the_sufficient_statistics(self):
        """Order of observation cannot change the ratio -- only the counts can.
        This is what lets a cached corpus be relabelled from (n, k) alone."""
        assert sprt_log_likelihood(20, 4, SHIPPED) == sprt_log_likelihood(20, 4, SHIPPED)
        assert sprt_log_likelihood(20, 4, SHIPPED) != sprt_log_likelihood(20, 5, SHIPPED)

    def test_it_stays_finite_at_large_counts(self):
        value = sprt_log_likelihood(10 ** 6, 10 ** 5, SHIPPED)

        assert math.isfinite(value)

    def test_a_neutral_alternative_carries_no_evidence(self):
        neutral = SPRTConfig(p1_factor=1.0)

        assert sprt_log_likelihood(50, 7, neutral) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Stopping rules
# --------------------------------------------------------------------------


class TestStoppingRules:

    def test_nothing_is_decided_before_the_first_episode(self):
        assert sprt_decision_for_counts(0, 0, SHIPPED) is None

    def test_the_zero_failure_rule_fires_exactly_at_the_required_budget(self):
        assert sprt_decision_for_counts(58, 0, SHIPPED) is None
        assert sprt_decision_for_counts(59, 0, SHIPPED) == SPRTDecision(
            ScenarioTestResult.SAFE, CP_K0
        )

    def test_the_zero_failure_rule_keeps_holding_afterwards(self):
        for n in (59, 60, 79, 80, 200):
            assert sprt_decision_for_counts(n, 0, SHIPPED).mode == CP_K0

    def test_three_consecutive_failures_are_terminal(self):
        assert sprt_decision_for_counts(1, 1, SHIPPED) is None
        assert sprt_decision_for_counts(2, 2, SHIPPED) is None
        assert sprt_decision_for_counts(3, 3, SHIPPED) == SPRTDecision(
            ScenarioTestResult.UNSAFE, WALD_UNSAFE
        )

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_the_cap_condemns_any_design_with_a_failure(self, k):
        """At n = 80 the CP bound for k >= 1 exceeds the target rate, so the cap
        can only clear a design that never failed."""
        assert cp_upper_bound(k, 80, ALPHA) > SHIPPED.P
        assert sprt_decision_for_counts(80, k, SHIPPED) == SPRTDecision(
            ScenarioTestResult.UNSAFE, CAP_UNSAFE
        )

    def test_the_zero_failure_rule_outranks_the_cap(self):
        assert sprt_decision_for_counts(80, 0, SHIPPED).mode == CP_K0

    def test_the_undecided_band_at_the_last_episode_before_the_cap(self):
        undecided = [k for k in range(80) if sprt_decision_for_counts(79, k, SHIPPED) is None]

        assert undecided == [1, 2, 3, 4, 5]

    def test_the_wald_thresholds_are_the_documented_wald_bounds(self):
        log_a = math.log((1.0 - SHIPPED.beta_err) / ALPHA)
        log_b = math.log(SHIPPED.beta_err / (1.0 - ALPHA))

        assert sprt_log_likelihood(3, 3, SHIPPED) <= log_b
        assert sprt_log_likelihood(2, 2, SHIPPED) > log_b
        assert log_b < 0.0 < log_a

    def test_the_wald_safe_rule_fires_when_it_is_reachable(self):
        """Raising the episode cap exposes the WALD_SAFE branch, which the
        shipped cap makes unreachable."""
        generous = SPRTConfig(global_max_iters=10_000)

        assert sprt_decision_for_counts(134, 1, generous) is None
        assert sprt_decision_for_counts(135, 1, generous) == SPRTDecision(
            ScenarioTestResult.SAFE, WALD_SAFE
        )

    def test_the_cp_cap_rule_clears_a_design_when_the_bound_is_tight_enough(self):
        """CP_CAP needs a cap reached while the bound still sits under the target
        rate -- a looser target with an early cap reaches it."""
        loose = SPRTConfig(P=0.2, global_max_iters=30)

        assert cp_upper_bound(1, 30, ALPHA) <= loose.P
        assert sprt_decision_for_counts(30, 1, loose) == SPRTDecision(
            ScenarioTestResult.SAFE, CP_CAP
        )
        assert sprt_decision_for_counts(30, 3, loose) == SPRTDecision(
            ScenarioTestResult.UNSAFE, CAP_UNSAFE
        )

    def test_a_decision_is_a_function_of_the_counts_alone(self):
        for n, k in [(10, 1), (59, 0), (80, 2), (3, 3)]:
            assert sprt_decision_for_counts(n, k, SHIPPED) == sprt_decision_for_counts(
                n, k, SHIPPED
            )

    @pytest.mark.parametrize("n,k", [(2, 3), (-1, 0)])
    def test_impossible_counts_propagate_the_likelihood_error(self, n, k):
        with pytest.raises(ValueError, match="Invalid SPRT counts"):
            sprt_decision_for_counts(n, k, SHIPPED)


class TestThresholdBoundariesAreInclusive:
    """Every stopping comparison is non-strict, so a run landing exactly on a
    threshold stops rather than drawing another episode. The equality is not
    hypothetical -- these configurations hit it exactly in floating point, and
    each test asserts that precondition before checking the verdict.
    """

    def test_the_unsafe_threshold_decides_on_equality(self):
        # log_b = log(0.2375 / 0.95) = log(1/4), and two failures at
        # p1_factor = 0.5 land on 2 * log(1/2) -- the same double.
        cfg = SPRTConfig(beta_err=0.2375, global_max_iters=10 ** 6)
        log_b = math.log(cfg.beta_err / (1.0 - (1.0 - cfg.C)))

        assert sprt_log_likelihood(2, 2, cfg) == log_b, "precondition: exact hit"
        assert sprt_decision_for_counts(2, 2, cfg) == SPRTDecision(
            ScenarioTestResult.UNSAFE, WALD_UNSAFE
        )

    def test_the_safe_threshold_decides_on_equality(self):
        alpha = 0.10
        probe = SPRTConfig(C=1.0 - alpha)
        # Place log_a exactly on the likelihood after a single clean episode.
        cfg = SPRTConfig(
            C=1.0 - alpha,
            beta_err=1.0 - alpha * math.exp(sprt_log_likelihood(1, 0, probe)),
            global_max_iters=10 ** 6,
        )
        log_a = math.log((1.0 - cfg.beta_err) / (1.0 - cfg.C))

        assert sprt_log_likelihood(1, 0, cfg) == log_a, "precondition: exact hit"
        assert sprt_decision_for_counts(1, 0, cfg) == SPRTDecision(
            ScenarioTestResult.SAFE, WALD_SAFE
        )

    def test_the_capped_bound_clears_a_design_sitting_on_the_target_rate(self):
        """`upper <= p0`: a bound landing exactly on the target failure rate is
        good enough to pass, not a hair short of it."""
        confidence = 0.95
        target = cp_upper_bound(1, 30, 1.0 - confidence)
        cfg = SPRTConfig(P=target, C=confidence, global_max_iters=30)

        assert cp_upper_bound(1, 30, 1.0 - cfg.C) == cfg.P, "precondition: exact hit"
        assert sprt_decision_for_counts(30, 1, cfg) == SPRTDecision(
            ScenarioTestResult.SAFE, CP_CAP
        )

    def test_the_zero_failure_budget_decides_on_equality(self):
        """`n >= min_iters_k0`: the budget episode itself decides."""
        budget = required_min_iters_k0(SHIPPED.P, ALPHA)

        assert sprt_decision_for_counts(budget - 1, 0, SHIPPED) is None
        assert sprt_decision_for_counts(budget, 0, SHIPPED).mode == CP_K0


def _reachable_stopping_states(cfg: SPRTConfig):
    """Every (n, k) a live run can stop at: breadth-first over paths that have
    not already stopped."""
    stops = {}
    frontier = [(0, 0)]
    seen = {(0, 0)}
    while frontier:
        nxt = []
        for n, k in frontier:
            for step in (0, 1):
                state = (n + 1, k + step)
                if state in seen:
                    continue
                seen.add(state)
                decision = sprt_decision_for_counts(state[0], state[1], cfg)
                if decision is None:
                    nxt.append(state)
                else:
                    stops[state] = decision
        frontier = nxt
    return stops


class TestShippedConfigurationReachability:
    """What the production settings can actually produce. These numbers describe
    every label in the corpus; a change to `SPRTConfig` defaults that moves them
    silently relabels past work."""

    def test_the_shipped_defaults_are_unchanged(self):
        assert SHIPPED == SPRTConfig(
            P=0.05,
            C=0.95,
            p1_factor=0.5,
            beta_err=0.2,
            global_max_iters=80,
            batch_size=10,
        )

    def test_every_run_terminates_within_the_episode_cap(self):
        stops = _reachable_stopping_states(SHIPPED)

        assert max(n for n, _ in stops) == SHIPPED.global_max_iters

    def test_only_three_of_the_five_modes_are_reachable(self):
        """`WALD_SAFE` needs n = 135 and `CP_CAP` needs a looser target rate, so
        neither can appear under the shipped cap of 80."""
        modes = {decision.mode for decision in _reachable_stopping_states(SHIPPED).values()}

        assert modes == {CP_K0, WALD_UNSAFE, CAP_UNSAFE}
        assert WALD_SAFE not in modes
        assert CP_CAP not in modes

    def test_safe_is_only_ever_awarded_for_a_flawless_run(self):
        stops = _reachable_stopping_states(SHIPPED)
        safe = [state for state, d in stops.items() if d.label == ScenarioTestResult.SAFE]

        assert safe == [(59, 0)]

    def test_an_unsafe_verdict_needs_at_least_one_failure(self):
        stops = _reachable_stopping_states(SHIPPED)
        unsafe = [state for state, d in stops.items() if d.label == ScenarioTestResult.UNSAFE]

        assert min(k for _, k in unsafe) == 1
        assert all(k >= 1 for _, k in unsafe)

    def test_the_early_unsafe_exit_needs_three_failures(self):
        stops = _reachable_stopping_states(SHIPPED)
        wald = [state for state, d in stops.items() if d.mode == WALD_UNSAFE]

        assert min(k for _, k in wald) == 3
        assert min(n for n, _ in wald) == 3

    def test_the_cap_exit_only_happens_at_the_cap(self):
        stops = _reachable_stopping_states(SHIPPED)
        capped = [state for state, d in stops.items() if d.mode == CAP_UNSAFE]

        assert {n for n, _ in capped} == {80}
        assert sorted(k for _, k in capped) == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------
# Sequential state
# --------------------------------------------------------------------------


def _surviving_path(n_target: int, k_target: int, cfg: SPRTConfig):
    """An observation order reaching `(n_target, k_target)` without stopping.

    Only some orderings of a given (n, k) are reachable: three failures in a row
    end the run at n = 3, and a clean prefix of 59 ends it at CP_K0. Failures are
    placed as early as the rules allow, which is what a live run that survives to
    these counts must look like.
    """
    n = k = 0
    path = []
    while (n, k) != (n_target, k_target):
        for candidate in (ScenarioTestResult.UNSAFE, ScenarioTestResult.SAFE):
            is_failure = candidate is ScenarioTestResult.UNSAFE
            if is_failure and k >= k_target:
                continue
            if not is_failure and (n - k) >= (n_target - k_target):
                continue
            step = (n + 1, k + int(is_failure))
            if step == (n_target, k_target) or sprt_decision_for_counts(*step, cfg) is None:
                path.append(candidate)
                n, k = step
                break
        else:
            raise AssertionError(f"({n_target}, {k_target}) is not reachable")
    return path


class TestSPRTState:

    def test_a_fresh_state_holds_no_evidence(self):
        state = SPRTState(SHIPPED)

        assert (state.n, state.k) == (0, 0)
        assert state.log_likelihood == 0.0
        assert state.decision is None

    def test_a_clean_episode_advances_only_the_trial_count(self):
        state = SPRTState(SHIPPED)

        state.observe(ScenarioTestResult.SAFE)

        assert (state.n, state.k) == (1, 0)

    def test_a_failure_advances_both_counts(self):
        state = SPRTState(SHIPPED)

        state.observe(ScenarioTestResult.UNSAFE)

        assert (state.n, state.k) == (1, 1)

    def test_the_running_likelihood_tracks_the_pure_function(self):
        state = SPRTState(SHIPPED)

        for result in [ScenarioTestResult.SAFE] * 7 + [ScenarioTestResult.UNSAFE] * 2:
            state.observe(result)
            assert state.log_likelihood == sprt_log_likelihood(state.n, state.k, SHIPPED)

    def test_stepping_agrees_with_the_rule_at_every_prefix(self):
        """The sequential state machine must not diverge from a direct
        evaluation of the stopping rules -- that equivalence is what lets a
        cached corpus be relabelled without re-running episodes."""
        state = SPRTState(SHIPPED)
        sequence = [ScenarioTestResult.SAFE] * 30 + [ScenarioTestResult.UNSAFE] + [ScenarioTestResult.SAFE] * 60

        for result in sequence:
            returned = state.observe(result)
            assert returned == sprt_decision_for_counts(state.n, state.k, SHIPPED)
            if returned is not None:
                break

        assert (state.n, state.k) == (80, 1)
        assert returned.mode == CAP_UNSAFE

    def test_observing_past_a_decision_is_refused(self):
        state = SPRTState(SHIPPED)
        for _ in range(3):
            state.observe(ScenarioTestResult.UNSAFE)

        with pytest.raises(RuntimeError, match="after an SPRT decision"):
            state.observe(ScenarioTestResult.SAFE)

    def test_a_state_resumed_at_a_decision_refuses_the_next_observation(self):
        with pytest.raises(RuntimeError, match="after an SPRT decision"):
            SPRTState(SHIPPED, n=59, k=0).observe(ScenarioTestResult.SAFE)

    @pytest.mark.parametrize("bogus", ["UNSAFE", None, 2, -1])
    def test_an_unrecognised_outcome_is_refused(self, bogus):
        with pytest.raises(ValueError, match="Unknown test result"):
            SPRTState(SHIPPED).observe(bogus)

    @pytest.mark.parametrize("n,k", [(0, 0), (30, 1), (58, 0), (79, 5)])
    def test_a_resumed_state_decides_exactly_as_a_replayed_one(self, n, k):
        """Warm-starting from sufficient statistics must be indistinguishable
        from having observed the episodes."""
        replayed = SPRTState(SHIPPED)
        for result in _surviving_path(n, k, SHIPPED):
            replayed.observe(result)

        resumed = SPRTState(SHIPPED, n=n, k=k)

        assert (replayed.n, replayed.k) == (resumed.n, resumed.k)
        assert replayed.decision == resumed.decision
        assert replayed.log_likelihood == resumed.log_likelihood

    def test_the_stopping_point_depends_on_the_order_of_observations(self):
        """The verdict is a function of (n, k), but *when* the run stops is not.
        Three failures up front stop at n = 3; the same three spread out run to
        the cap."""
        early = SPRTState(SHIPPED)
        for result in [ScenarioTestResult.UNSAFE] * 3:
            decision = early.observe(result)

        late = SPRTState(SHIPPED)
        sequence = [ScenarioTestResult.SAFE] * 30 + [ScenarioTestResult.UNSAFE] * 3 + [ScenarioTestResult.SAFE] * 60
        for result in sequence:
            late_decision = late.observe(result)
            if late_decision is not None:
                break

        assert decision.label == late_decision.label == ScenarioTestResult.UNSAFE
        assert (early.n, decision.mode) == (3, WALD_UNSAFE)
        assert (late.n, late_decision.mode) == (80, CAP_UNSAFE)

    def test_each_state_carries_its_own_counts(self):
        first = SPRTState(SHIPPED)
        second = SPRTState(SHIPPED)

        first.observe(ScenarioTestResult.UNSAFE)

        assert (second.n, second.k) == (0, 0)


# --------------------------------------------------------------------------
# The production tester
# --------------------------------------------------------------------------


class _SequenceSim:
    """Replays a fixed episode list and records the batch sizes requested."""

    def __init__(self, results):
        self.results = list(results)
        self.position = 0
        self.batches = []

    def simulate(self, _scenario, _base_cfg, _design, num_eps):
        self.batches.append(num_eps)
        batch = self.results[self.position:self.position + num_eps]
        self.position += num_eps
        return batch


def _episode(safe: bool) -> EpisodeResult:
    result = EpisodeResult()
    result.result = {"safe": safe}
    return result


def _violation(episode: EpisodeResult) -> ScenarioTestResult:
    return ScenarioTestResult.SAFE if episode.result["safe"] else ScenarioTestResult.UNSAFE


def _tester(sequence, cfg=SHIPPED):
    sim = _SequenceSim(_episode(value) for value in sequence)
    return ScenarioTester(sim, "intersection1", {}, cfg, _violation), sim


class TestScenarioTesterConstants:
    """The tester recomputes the Wald thresholds itself instead of calling the
    shared helpers. Pin them equal, or online and offline labelling drift
    apart."""

    def test_the_wald_thresholds_match_the_shared_rule(self):
        tester, _ = _tester([True])

        assert tester.logA == math.log((1.0 - SHIPPED.beta_err) / ALPHA)
        assert tester.logB == math.log(SHIPPED.beta_err / (1.0 - ALPHA))

    def test_the_zero_failure_budget_matches_the_shared_helper(self):
        tester, _ = _tester([True])

        assert tester.min_iters_k0 == required_min_iters_k0(SHIPPED.P, ALPHA)

    def test_the_alternative_rate_matches_the_shared_rule(self):
        tester, _ = _tester([True])

        assert tester.p1 == SHIPPED.p1_factor * SHIPPED.P
        assert tester.p0 == SHIPPED.P
        assert tester.alpha_err == ALPHA

    def test_the_configuration_is_copied_not_aliased(self):
        cfg = SPRTConfig()
        tester, _ = _tester([True], cfg=cfg)

        cfg.global_max_iters = 5

        assert tester.sprt_cfg.global_max_iters == 80


class TestScenarioTesterLabelling:

    def test_a_flawless_design_is_cleared_by_the_zero_failure_rule(self):
        tester, _ = _tester([True] * 200)

        label, info = tester.sprt_label({})

        assert label == ScenarioTestResult.SAFE
        assert (info.mode, info.n, info.k) == (CP_K0, 59, 0)

    def test_three_early_failures_stop_after_one_batch(self):
        tester, sim = _tester([False] * 3 + [True] * 200)

        label, info = tester.sprt_label({})

        assert label == ScenarioTestResult.UNSAFE
        assert (info.mode, info.n, info.k) == (WALD_UNSAFE, 3, 3)
        assert sim.batches == [SHIPPED.batch_size]

    def test_a_single_failure_runs_to_the_cap_and_condemns_the_design(self):
        tester, _ = _tester([False] + [True] * 200)

        label, info = tester.sprt_label({})

        assert label == ScenarioTestResult.UNSAFE
        assert (info.mode, info.n, info.k) == (CAP_UNSAFE, 80, 1)

    def test_episodes_are_drawn_in_whole_batches(self):
        """The decision lands mid-batch, so more episodes are simulated than the
        verdict counts. `raw` holds everything that was run; `n` holds what was
        scored."""
        tester, sim = _tester([True] * 200)

        _, info = tester.sprt_label({})

        assert info.n == 59
        assert len(info.raw) == 60
        assert sim.batches == [10] * 6

    def test_the_batch_size_is_honoured(self):
        tester, sim = _tester([True] * 200, cfg=SPRTConfig(batch_size=25))

        _, info = tester.sprt_label({})

        assert sim.batches == [25] * 3
        assert info.n == 59

    def test_the_online_label_matches_the_offline_replay_of_the_same_episodes(self):
        """The primitive-level counterpart of the corpus replay check: identical
        episodes, identical verdict, identical counts."""
        for sequence in (
            [True] * 200,
            [False] * 3 + [True] * 200,
            [False] + [True] * 200,
            [True] * 30 + [False] * 3 + [True] * 200,
        ):
            tester, _ = _tester(sequence)
            online_label, online_info = tester.sprt_label({})
            offline = offline_sprt_label(
                (_episode(value) for value in sequence), SHIPPED, _violation
            )

            assert offline.label == online_label
            assert offline.mode == online_info.mode
            assert offline.episodes_used == online_info.n
            assert offline.failures == online_info.k


class TestScenarioTesterCpOnlyPath:
    """`cp_only_label` is the non-sequential fallback: it stops at the first
    failure rather than accumulating evidence."""

    def test_a_flawless_design_is_cleared_at_the_zero_failure_budget(self):
        tester, _ = _tester([True] * 200)

        label, info = tester.cp_only_label({})

        assert label == ScenarioTestResult.SAFE
        assert (info.mode, info.n, info.k) == (CP_K0, 59, 0)

    def test_the_first_failure_is_terminal(self):
        tester, _ = _tester([True] * 5 + [False] + [True] * 200)

        label, info = tester.cp_only_label({})

        assert label == ScenarioTestResult.UNSAFE
        assert (info.mode, info.n, info.k) == (OBS_FAIL, 6, 1)

    def test_it_is_strictly_less_tolerant_than_the_sequential_test(self):
        """One failure is fatal here but survivable under the SPRT, which is the
        whole reason the sequential path exists."""
        sequence = [True] * 5 + [False] + [True] * 200

        cp_label, _ = _tester(sequence)[0].cp_only_label({})
        sprt_label, sprt_info = _tester(sequence)[0].sprt_label({})

        assert cp_label == ScenarioTestResult.UNSAFE
        assert sprt_label == ScenarioTestResult.UNSAFE
        assert sprt_info.n > 6, "the SPRT keeps gathering evidence after a failure"


class TestScenarioTesterResilience:

    def test_a_transient_simulator_failure_is_retried(self, monkeypatch):
        import source.design_exploration.commons.scenario_labeling as labeling

        monkeypatch.setattr(labeling.time, "sleep", lambda _seconds: None)

        class _Flaky:
            """The retry path rebuilds the interface, so the failure budget has
            to live on the class rather than the instance."""

            output_folder = "/tmp"
            algo_version = "v1"
            num_carla_instances = 1
            calls = 0

            def __init__(self, **_kwargs):
                pass

            def simulate(self, _scenario, _cfg, _design, num_eps):
                _Flaky.calls += 1
                if _Flaky.calls == 1:
                    raise RuntimeError("carla died")
                return [_episode(True) for _ in range(num_eps)]

            def close(self):
                pass

        original = _Flaky()
        tester = ScenarioTester(original, "s", {}, SPRTConfig(batch_size=2), _violation)

        assert len(tester._episodes({}, 2)) == 2
        assert _Flaky.calls == 2, "the batch is re-run, not skipped"
        assert tester.sim is not original, "the interface is rebuilt on restart"

    def test_a_persistent_simulator_failure_is_surfaced(self, monkeypatch):
        import source.design_exploration.commons.scenario_labeling as labeling

        monkeypatch.setattr(labeling.time, "sleep", lambda _seconds: None)

        class _Dead:
            output_folder = "/tmp"
            algo_version = "v1"
            num_carla_instances = 1

            def __init__(self, **_kwargs):
                pass

            def simulate(self, *_args):
                raise RuntimeError("carla died")

            def close(self):
                pass

        tester = ScenarioTester(_Dead(), "s", {}, SPRTConfig(batch_size=2), _violation)

        with pytest.raises(RuntimeError, match="carla died"):
            tester._episodes({}, 2)
