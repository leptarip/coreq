# Copyright (c) 2026 278097159+leptarip@users.noreply.github.com
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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.design_exploration.audit.audit3_portfolio import (
    build_frozen_stage_plans,
    build_reuse_assignments,
    run_parallel_cached_portfolio_stage,
    should_run_stage,
    validate_declaration,
)
from source.design_exploration.commons.episodes.episode_result import EpisodeResult


SCENARIOS = ("s1", "s2")


def _declaration(delta2=0.004):
    return {
        "method": "bonferroni",
        "global_delta": 0.01,
        "cache_reuse": "matching_design_id",
        "stages": [
            {"id": "primary", "tau": 0.5, "delta": 0.006, "n": 4,
             "rng_seed": 11, "run_if": "always"},
            {"id": "fallback", "tau": 0.8, "delta": delta2, "n": 3,
             "rng_seed": 12,
             "run_if": {"stage": "primary", "passed": False}},
        ],
    }


def _episode(uid, violation=False, source="primary"):
    return {
        "ep_uid": uid,
        "requested_seed": 101,
        "seed": 101,
        "min_d": 0.4 if violation else 0.8,
        "violation": violation,
        "malformed": False,
        "malformed_reason": None,
        "reused": False,
        "planned_seed": 101,
        "planned_seed_executed": True,
        "reused_from_stage": source,
    }


def test_declaration_supports_unequal_bonferroni_and_conditional_gate():
    stages = validate_declaration(_declaration(), scenario_names=SCENARIOS)
    assert [stage["delta"] for stage in stages] == [0.006, 0.004]
    assert should_run_stage(stages[1], {"primary": {"passed": False}})
    assert not should_run_stage(stages[1], {"primary": {"passed": True}})
    assert not should_run_stage(
        stages[1], {"primary": {"status": "skipped", "passed": False}}
    )


def test_declaration_rejects_overspent_or_unknown_method():
    with pytest.raises(ValueError, match="exceeds global_delta"):
        validate_declaration(_declaration(delta2=0.005), scenario_names=SCENARIOS)
    bad = _declaration()
    bad["method"] = "sidak"
    with pytest.raises(ValueError, match="bonferroni"):
        validate_declaration(bad, scenario_names=SCENARIOS)


def test_gate_chain_propagates_a_skipped_dependency_without_crashing():
    declaration = _declaration()
    declaration["stages"].append({
        "id": "third", "tau": 0.9, "delta": 1e-4, "n": 3,
        "rng_seed": 13,
        "run_if": {"stage": "fallback", "passed": False},
    })
    declaration["stages"][0]["delta"] = 0.005
    declaration["stages"][1]["delta"] = 0.004
    stages = validate_declaration(declaration, scenario_names=SCENARIOS)
    outcomes = {"primary": {"passed": True}}
    assert not should_run_stage(stages[1], outcomes)
    outcomes["fallback"] = {"status": "skipped", "passed": False}
    assert not should_run_stage(stages[2], outcomes)


def test_all_stage_plans_are_fixed_and_seeds_are_globally_unique():
    stages = validate_declaration(_declaration(), scenario_names=SCENARIOS)
    scores = np.array([0.1, 0.5, 0.8, 0.9])
    summaries1, plans1 = build_frozen_stage_plans(
        scores=scores, stages=stages, scenario_names=SCENARIOS
    )
    summaries2, plans2 = build_frozen_stage_plans(
        scores=scores, stages=stages, scenario_names=SCENARIOS
    )
    assert summaries1 == summaries2
    assert plans1 == plans2
    assert summaries1[0]["accepted_count"] == 3
    assert summaries1[1]["accepted_count"] == 2
    seeds = [
        seed for plan in plans1.values() for trial in plan
        for seed in trial["seeds"].values()
    ]
    assert len(seeds) == len(set(seeds))


def test_cache_reuse_requires_exact_design_and_scenario_and_is_fifo():
    prior = [
        {"stage_id": "primary", "gidx": 4,
         "scenarios": {"s1": _episode("u1"), "s2": _episode("u2")}},
        {"stage_id": "primary", "gidx": 4,
         "scenarios": {"s1": _episode("u3"), "s2": _episode("u4")}},
    ]
    plan = [
        {"trial_index": 0, "gidx": 4, "seeds": {"s1": 201, "s2": 202}},
        {"trial_index": 1, "gidx": 5, "seeds": {"s1": 203, "s2": 204}},
        {"trial_index": 2, "gidx": 4, "seeds": {"s1": 205, "s2": 206}},
        {"trial_index": 3, "gidx": 4, "seeds": {"s1": 207, "s2": 208}},
    ]
    reused = build_reuse_assignments(
        trial_plan=plan, scenario_names=SCENARIOS, prior_trials=prior
    )
    assert set(reused) == {(0, "s1"), (0, "s2"), (2, "s1"), (2, "s2")}
    assert reused[(0, "s1")]["ep_uid"] == "u1"
    assert reused[(2, "s1")]["ep_uid"] == "u3"
    assert reused[(0, "s1")]["planned_seed"] == 201
    assert reused[(0, "s1")]["planned_seed_executed"] is False


class _Manager:
    def __init__(self, scenario="s"):
        self.scenario = scenario
        self.count = 0
        self.flushed = False

    def append_one(self, _gidx, _episode, _meta):
        uid = f"{self.scenario}-fresh-{self.count}"
        self.count += 1
        return uid

    def flush(self):
        self.flushed = True


class _Pool:
    def __init__(self):
        self.active = {}
        self.max_active = 0
        self.fatal_error = None
        self.alive_workers = 2

    def submit(self, *, job_id, scenario_name, scenario_cfg, design, seed, num_episodes):
        assert scenario_cfg == {} and num_episodes == 1 and design
        self.active[job_id] = (scenario_name, seed)
        self.max_active = max(self.max_active, len(self.active))
        return job_id

    def poll(self):
        job_id, (_scenario, seed) = self.active.popitem()
        episode = EpisodeResult()
        episode.result["safety_m"]["min_d"] = 0.8
        episode.result["meta"] = {"requested_seed": seed, "simulator_seed": seed}
        return [SimpleNamespace(
            job_id=job_id, status="done", payload=[episode], error=None,
            attempts=1, worker_id=0,
        )]


class _BatchPool(_Pool):
    def poll(self):
        records = []
        while self.active:
            job_id, (_scenario, seed) = self.active.popitem()
            episode = EpisodeResult()
            episode.result["safety_m"]["min_d"] = 0.8
            episode.result["meta"] = {
                "requested_seed": seed, "simulator_seed": seed,
            }
            records.append(SimpleNamespace(
                job_id=job_id, status="done", payload=[episode], error=None,
                attempts=1, worker_id=0,
            ))
        return records


def test_fully_cached_stage_does_not_require_worker_pool():
    plan = [{"trial_index": 0, "gidx": 0, "seeds": {"s1": 10, "s2": 11}}]
    cached = {
        (0, "s1"): {**_episode("u1"), "planned_seed": 10},
        (0, "s2"): {**_episode("u2"), "planned_seed": 11},
    }
    result, usage = run_parallel_cached_portfolio_stage(
        stage_id="fallback", threshold=0.8, accepted_count=1,
        trial_plan=plan, scenario_names=SCENARIOS,
        scenario_cfgs={"s1": {}, "s2": {}}, grid=np.array([[1.0]]),
        design_keys=("x",), pool=None,
        epmans={"s1": _Manager(), "s2": _Manager()},
        P=1.0, alpha_frac=1.0, delta=0.01, d_safe=0.5,
        max_in_flight=2, reused_episode_results=cached,
    )
    assert usage == {}
    assert result["scheduler"]["submitted_jobs"] == 0
    assert result["cache_reuse"]["reused_episode_jobs"] == 2
    assert result["trials"][0]["scenarios"]["s1"]["ep_uid"] == "u1"


def test_reused_label_must_match_d_safe():
    plan = [{"trial_index": 0, "gidx": 0, "seeds": {"s1": 10, "s2": 11}}]
    bad = _episode("u1")
    bad["min_d"] = 0.4
    cached = {(0, "s1"): bad, (0, "s2"): _episode("u2")}
    with pytest.raises(ValueError, match="d_safe"):
        run_parallel_cached_portfolio_stage(
            stage_id="fallback", threshold=0.8, accepted_count=1,
            trial_plan=plan, scenario_names=SCENARIOS,
            scenario_cfgs={"s1": {}, "s2": {}}, grid=np.array([[1.0]]),
            design_keys=("x",), pool=None,
            epmans={"s1": _Manager(), "s2": _Manager()},
            P=1.0, alpha_frac=1.0, delta=0.01, d_safe=0.5,
            max_in_flight=2, reused_episode_results=cached,
        )


def test_partial_cache_submits_only_missing_jobs_to_bounded_pool():
    plan = [
        {"trial_index": 0, "gidx": 0, "seeds": {"s1": 10, "s2": 11}},
        {"trial_index": 1, "gidx": 1, "seeds": {"s1": 12, "s2": 13}},
    ]
    cached = {(0, "s1"): _episode("cached-s1")}
    pool = _Pool()
    result, usage = run_parallel_cached_portfolio_stage(
        stage_id="fallback", threshold=0.8, accepted_count=2,
        trial_plan=plan, scenario_names=SCENARIOS,
        scenario_cfgs={"s1": {}, "s2": {}}, grid=np.array([[1.0], [2.0]]),
        design_keys=("x",), pool=pool,
        epmans={"s1": _Manager("s1"), "s2": _Manager("s2")},
        P=1.0, alpha_frac=1.0, delta=0.01, d_safe=0.5,
        max_in_flight=2, reused_episode_results=cached, poll_interval_s=0.0,
    )
    assert pool.max_active == 2
    assert result["scheduler"]["submitted_jobs"] == 3
    assert result["cache_reuse"] == {
        "policy": "matching_design_id",
        "reused_episode_jobs": 1,
        "fresh_episode_jobs": 3,
        "unique_uid_within_stage": True,
    }
    assert sum(usage.values()) == 3
    assert [trial["trial_index"] for trial in result["trials"]] == [0, 1]
    assert [trial["gidx"] for trial in result["trials"]] == [0, 1]


def test_checkpoint_callback_runs_once_per_poll_batch():
    plan = [{"trial_index": 0, "gidx": 0, "seeds": {"s1": 10, "s2": 11}}]
    batches = []
    run_parallel_cached_portfolio_stage(
        stage_id="primary", threshold=0.8, accepted_count=1,
        trial_plan=plan, scenario_names=SCENARIOS,
        scenario_cfgs={"s1": {}, "s2": {}}, grid=np.array([[1.0]]),
        design_keys=("x",), pool=_BatchPool(),
        epmans={"s1": _Manager("s1"), "s2": _Manager("s2")},
        P=1.0, alpha_frac=1.0, delta=0.01, d_safe=0.5,
        max_in_flight=2, poll_interval_s=0.0,
        checkpoint_callback=lambda batch: batches.append(dict(batch)),
    )
    assert len(batches) == 1
    assert set(batches[0]) == {(0, "s1"), (0, "s2")}


class _FailingPool(_Pool):
    def __init__(self, successful_jobs=0):
        super().__init__()
        self.successful_jobs = successful_jobs

    def poll(self):
        if self.successful_jobs:
            self.successful_jobs -= 1
            return super().poll()
        job_id, _ = self.active.popitem()
        return [SimpleNamespace(job_id=job_id, status="failed", payload=None,
                                error="terminal simulator failure", attempts=3, worker_id=0)]


class _MalformedPool(_Pool):
    def poll(self):
        records = super().poll()
        records[0].payload[0].result["safety_m"] = {}
        return records


@pytest.fixture
def fresh_stage():
    return dict(
        stage_id="primary", threshold=0.8, accepted_count=2,
        trial_plan=[
            {"trial_index": 0, "gidx": 0, "seeds": {"s1": 10, "s2": 11}},
            {"trial_index": 1, "gidx": 1, "seeds": {"s1": 12, "s2": 13}},
        ],
        scenario_names=SCENARIOS, scenario_cfgs={"s1": {}, "s2": {}},
        grid=np.array([[1.0], [2.0]]), design_keys=("x",),
        epmans={s: _Manager(s) for s in SCENARIOS},
        P=1.0, alpha_frac=1.0, delta=0.01, d_safe=0.5,
        max_in_flight=1, poll_interval_s=0.0,
    )


def test_terminal_failure_flushes_partial_episode_stores(fresh_stage):
    with pytest.raises(RuntimeError, match="terminal simulator failure"):
        run_parallel_cached_portfolio_stage(pool=_FailingPool(), **fresh_stage)
    assert all(manager.flushed for manager in fresh_stage["epmans"].values())


def test_resume_after_failure_submits_only_missing_jobs(fresh_stage):
    durable = {}
    with pytest.raises(RuntimeError, match="terminal simulator failure"):
        run_parallel_cached_portfolio_stage(
            pool=_FailingPool(successful_jobs=2), checkpoint_callback=durable.update, **fresh_stage
        )
    assert set(durable) == {(0, "s1"), (0, "s2")}
    result, usage = run_parallel_cached_portfolio_stage(
        pool=_Pool(), resumed_episode_results=durable, **fresh_stage
    )
    assert result["scheduler"]["submitted_jobs"] == 2
    assert sum(usage.values()) == 2
    assert [trial["trial_index"] for trial in result["trials"]] == [0, 1]
    for trial, planned in zip(result["trials"], fresh_stage["trial_plan"]):
        for scenario in SCENARIOS:
            assert trial["scenarios"][scenario]["seed"] == planned["seeds"][scenario]
    completed = {(trial["trial_index"], scenario): trial["scenarios"][scenario]
                 for trial in result["trials"] for scenario in SCENARIOS}
    replayed, replay_usage = run_parallel_cached_portfolio_stage(
        pool=None, resumed_episode_results=completed, **fresh_stage
    )
    assert replayed["scheduler"]["submitted_jobs"] == 0
    assert replayed["trials"] == result["trials"]
    assert replay_usage == {}


def test_missing_min_d_is_counted_as_a_violation(fresh_stage):
    result, usage = run_parallel_cached_portfolio_stage(pool=_MalformedPool(), **fresh_stage)
    assert result["n"] == result["k"] == 2
    assert result["malformed_episodes"] == 4
    assert sum(usage.values()) == 4
    for trial in result["trials"]:
        assert trial["malformed"] and trial["violation"]
        for episode in trial["scenarios"].values():
            assert episode["min_d"] is None
            assert episode["violation"] and episode["malformed"]
            assert episode["malformed_reason"] == "missing_or_non_finite_min_d"
