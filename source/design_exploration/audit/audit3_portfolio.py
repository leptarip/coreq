# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Frozen, staged portfolio-audit primitives used by Audit 3.

The module is deliberately simulator-free at import time.  It provides the
pure validation/planning operations used by ``reproduction/phase1`` and a
worker-pool executor whose pool and episode managers are injected by callers.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from source.design_exploration.audit.audit_common import pool_record_to_episode_result
from source.design_exploration.audit.trial_planning import build_trial_plan
from source.design_exploration.commons.episodes.episode_manager import EpisodeMeta
from source.design_exploration.commons.scenario_labeling import cp_upper_bound
from source.design_exploration.simulator_interface.worker_protocol import verified_simulator_seed

if TYPE_CHECKING:
    from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
    from source.design_exploration.simulator_interface.carla_worker_pool import CarlaWorkerPool


ALGO_VERSION = "audit_portfolio_parallel_v3"
ReuseKey = Tuple[int, str]
ResultKey = Tuple[int, str]


def validate_declaration(
    declaration: Mapping[str, object], *, scenario_names: Sequence[str]
) -> List[Dict[str, object]]:
    """Validate and normalize a predeclared staged Bonferroni protocol."""
    if declaration.get("method") != "bonferroni":
        raise ValueError("Audit 3 currently supports method='bonferroni' only")
    global_delta = float(declaration.get("global_delta", 0.0))
    if not 0.0 < global_delta < 1.0:
        raise ValueError("global_delta must be in (0, 1)")
    if declaration.get("cache_reuse", "matching_design_id") != "matching_design_id":
        raise ValueError("cache_reuse must be 'matching_design_id'")
    if not scenario_names or len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario names must be non-empty and unique")

    raw_stages = declaration.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("declaration.stages must be a non-empty list")
    stages: List[Dict[str, object]] = []
    seen: set[str] = set()
    allocated = 0.0
    for position, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"stage {position} must be an object")
        stage_id = str(raw.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stage_id) or stage_id in seen:
            raise ValueError(
                f"stage IDs must be unique filesystem-safe identifiers: {stage_id!r}"
            )
        seen.add(stage_id)
        tau = float(raw.get("tau", math.nan))
        delta = float(raw.get("delta", math.nan))
        n = int(raw.get("n", 0))
        rng_seed = int(raw.get("rng_seed", -1))
        if not math.isfinite(tau):
            raise ValueError(f"stage {stage_id}: tau must be finite")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"stage {stage_id}: delta must be in (0, 1)")
        if n <= 0 or rng_seed < 0:
            raise ValueError(f"stage {stage_id}: n must be positive and rng_seed non-negative")
        run_if = raw.get("run_if", "always")
        if position == 0 and run_if != "always":
            raise ValueError("the first stage must use run_if='always'")
        if run_if != "always":
            if not isinstance(run_if, Mapping):
                raise ValueError(f"stage {stage_id}: run_if must be 'always' or an object")
            dependency = str(run_if.get("stage", ""))
            if dependency not in seen or dependency == stage_id:
                raise ValueError(f"stage {stage_id}: run_if must reference an earlier stage")
            if not isinstance(run_if.get("passed"), bool):
                raise ValueError(f"stage {stage_id}: run_if.passed must be boolean")
            run_if = {"stage": dependency, "passed": bool(run_if["passed"])}
        allocated += delta
        stages.append(
            {
                "id": stage_id,
                "tau": tau,
                "delta": delta,
                "n": n,
                "rng_seed": rng_seed,
                "run_if": run_if,
            }
        )
    if allocated > global_delta + 1e-15:
        raise ValueError(
            f"Bonferroni allocation {allocated:.17g} exceeds global_delta {global_delta:.17g}"
        )
    return stages


def build_frozen_stage_plans(
    *,
    scores: np.ndarray,
    stages: Sequence[Mapping[str, object]],
    scenario_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    """Compute memberships and fix every stage's sampling plan up front."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite one-dimensional array")
    summaries: List[Dict[str, object]] = []
    plans: Dict[str, List[Dict[str, object]]] = {}
    all_seeds: set[int] = set()
    for stage in stages:
        stage_id = str(stage["id"])
        accepted = np.flatnonzero(values >= float(stage["tau"])).astype(np.int64)
        if accepted.size == 0:
            raise ValueError(f"stage {stage_id}: threshold accepts no designs")
        plan = build_trial_plan(
            accepted_idx=accepted,
            scenario_names=scenario_names,
            n=int(stage["n"]),
            rng_seed=int(stage["rng_seed"]),
        )
        seeds = {
            int(seed)
            for trial in plan
            for seed in dict(trial["seeds"]).values()
        }
        collision = all_seeds.intersection(seeds)
        if collision:
            raise ValueError(
                f"stage {stage_id}: episode seeds collide with an earlier stage; "
                "use distinct rng_seed values"
            )
        all_seeds.update(seeds)
        plans[stage_id] = plan
        summaries.append(
            {
                **dict(stage),
                "accepted_count": int(accepted.size),
                "accepted_idx": accepted.tolist(),
                "planned_unique_designs": len({int(t["gidx"]) for t in plan}),
                "planned_episode_jobs": len(plan) * len(scenario_names),
            }
        )
    return summaries, plans


def should_run_stage(
    stage: Mapping[str, object], outcomes: Mapping[str, Mapping[str, object]]
) -> bool:
    gate = stage.get("run_if", "always")
    if gate == "always":
        return True
    if not isinstance(gate, Mapping):
        raise ValueError(f"invalid run_if for stage {stage.get('id')!r}")
    dependency = str(gate["stage"])
    if dependency not in outcomes:
        raise ValueError(f"stage gate depends on unavailable result {dependency!r}")
    if outcomes[dependency].get("status") == "skipped":
        return False
    return bool(outcomes[dependency].get("passed", False)) == bool(gate["passed"])


def build_reuse_assignments(
    *,
    trial_plan: Sequence[Mapping[str, object]],
    scenario_names: Sequence[str],
    prior_trials: Sequence[Mapping[str, object]],
) -> Dict[ResultKey, Dict[str, object]]:
    """Assign cached episodes FIFO, only for an exact ``(gidx, scenario)`` match.

    Each episode UID is usable at most once in the new stage.  Reusing a result
    in another predeclared stage is allowed; Bonferroni validity does not require
    the stage statistics to be independent.
    """
    buckets: Dict[ReuseKey, Deque[Dict[str, object]]] = defaultdict(deque)
    seen_prior_uids: set[str] = set()
    for trial in prior_trials:
        gidx = int(trial["gidx"])
        source_stage = str(trial.get("stage_id", ""))
        scenarios = trial.get("scenarios")
        if not isinstance(scenarios, Mapping):
            raise ValueError("prior trial lacks scenario results")
        for scenario_name in scenario_names:
            raw = scenarios.get(scenario_name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"prior trial lacks scenario {scenario_name!r}")
            result = dict(raw)
            uid = str(result.get("ep_uid", ""))
            if not uid:
                raise ValueError("prior episode UID is missing")
            # With three or more stages, a stage may itself reference an episode
            # reused from an earlier stage. It remains one cache item, not two.
            if uid in seen_prior_uids:
                continue
            seen_prior_uids.add(uid)
            result["reused_from_stage"] = source_stage
            buckets[(gidx, scenario_name)].append(result)

    assigned: Dict[ResultKey, Dict[str, object]] = {}
    used_here: set[str] = set()
    for trial in trial_plan:
        trial_index = int(trial["trial_index"])
        gidx = int(trial["gidx"])
        seeds = trial["seeds"]
        for scenario_name in scenario_names:
            bucket = buckets[(gidx, scenario_name)]
            if not bucket:
                continue
            result = bucket.popleft()
            uid = str(result["ep_uid"])
            if uid in used_here:
                raise RuntimeError(f"cache assignment duplicated UID {uid}")
            used_here.add(uid)
            result.update(
                reused=True,
                planned_seed=int(seeds[scenario_name]),
                planned_seed_executed=False,
            )
            assigned[(trial_index, scenario_name)] = result
    return assigned


def _validate_result(
    result: Mapping[str, object], *, reused: bool, planned_seed: int, d_safe: float
) -> Dict[str, object]:
    checked = dict(result)
    uid = str(checked.get("ep_uid", ""))
    if not uid:
        raise ValueError("episode result has no UID")
    requested = int(checked.get("requested_seed", -1))
    actual = int(checked.get("seed", -1))
    if requested < 0 or actual != requested:
        raise ValueError("episode does not preserve a valid requested/simulator seed")
    if not reused:
        if requested != planned_seed or actual != requested:
            raise ValueError("fresh episode seed does not match the frozen plan")
    malformed = bool(checked.get("malformed", False))
    raw_min_d = checked.get("min_d")
    try:
        min_d = float(raw_min_d)
    except (TypeError, ValueError):
        min_d = None
    if malformed:
        if min_d is not None and math.isfinite(min_d):
            raise ValueError("malformed episode unexpectedly has a finite min_d")
        if not bool(checked.get("violation", False)):
            raise ValueError("missing/non-finite min_d must be conservatively marked unsafe")
        checked["min_d"] = None
    else:
        if min_d is None or not math.isfinite(min_d):
            raise ValueError("non-malformed episode must have a finite min_d")
        checked["min_d"] = min_d
        if bool(checked.get("violation")) != bool(min_d < float(d_safe)):
            raise ValueError("episode violation label disagrees with frozen d_safe")
    checked["reused"] = bool(reused)
    checked["planned_seed"] = int(planned_seed)
    checked["planned_seed_executed"] = not reused
    return checked


def run_parallel_cached_portfolio_stage(
    *,
    stage_id: str,
    threshold: float,
    accepted_count: int,
    trial_plan: Sequence[Mapping[str, object]],
    scenario_names: Sequence[str],
    scenario_cfgs: Mapping[str, Dict],
    grid: np.ndarray,
    design_keys: Sequence[str],
    pool: "Optional[CarlaWorkerPool]",
    epmans: Mapping[str, "EpisodeManager"],
    P: float,
    alpha_frac: float,
    delta: float,
    d_safe: float,
    max_in_flight: int,
    reused_episode_results: Optional[Mapping[ResultKey, Mapping[str, object]]] = None,
    resumed_episode_results: Optional[Mapping[ResultKey, Mapping[str, object]]] = None,
    checkpoint_callback: Optional[Callable[[Mapping[ResultKey, Mapping[str, object]]], None]] = None,
    poll_interval_s: float = 0.2,
) -> Tuple[Dict[str, object], Dict[int, int]]:
    """Execute one frozen stage with exact-design cache reuse and bounded concurrency."""
    scenario_names = list(scenario_names)
    if int(max_in_flight) <= 0:
        raise ValueError("max_in_flight must be positive")
    if set(scenario_cfgs) != set(scenario_names) or set(epmans) != set(scenario_names):
        raise ValueError("scenario_cfgs and epmans must match scenario_names exactly")
    plans: Dict[ResultKey, Dict[str, object]] = {}
    if [int(t["trial_index"]) for t in trial_plan] != list(range(len(trial_plan))):
        raise ValueError("trial indices must be contiguous and ordered from zero")
    for trial in trial_plan:
        ti, gidx = int(trial["trial_index"]), int(trial["gidx"])
        seeds = trial.get("seeds")
        if not isinstance(seeds, Mapping) or set(seeds) != set(scenario_names):
            raise ValueError(f"trial {ti} has invalid scenario seeds")
        if not 0 <= gidx < len(grid):
            raise ValueError(f"trial {ti} has out-of-range gidx {gidx}")
        for scenario in scenario_names:
            plans[(ti, scenario)] = {
                "trial_index": ti, "gidx": gidx, "scenario": scenario,
                "seed": int(seeds[scenario]),
            }

    episode_results: Dict[ResultKey, Dict[str, object]] = {}
    used_uids: set[str] = set()
    for source, expected_reused in (
        (reused_episode_results or {}, True),
        (resumed_episode_results or {}, None),
    ):
        for raw_key, raw in source.items():
            key = (int(raw_key[0]), str(raw_key[1]))
            planned = plans.get(key)
            if planned is None:
                raise ValueError(f"episode result is outside the frozen plan: {key!r}")
            if key in episode_results:
                # A durable checkpoint takes precedence only if it is identical.
                if dict(episode_results[key]) != dict(raw):
                    raise ValueError(f"checkpoint conflicts with frozen cache assignment: {key!r}")
                continue
            is_reused = bool(raw.get("reused", False)) if expected_reused is None else expected_reused
            checked = _validate_result(
                raw, reused=is_reused, planned_seed=int(planned["seed"]), d_safe=float(d_safe)
            )
            uid = str(checked["ep_uid"])
            if uid in used_uids:
                raise ValueError(f"stage duplicates episode UID {uid}")
            used_uids.add(uid)
            episode_results[key] = checked

    jobs = [meta for key, meta in plans.items() if key not in episode_results]
    if jobs and pool is None:
        raise ValueError("a worker pool is required for uncached jobs")
    submitted = 0
    completed = len(episode_results)
    active: Dict[str, Dict[str, object]] = {}
    sim_usage: Dict[int, int] = {}
    started = time.time()

    def submit_until_full() -> None:
        nonlocal submitted
        while submitted < len(jobs) and len(active) < int(max_in_flight):
            meta = jobs[submitted]
            gidx, scenario = int(meta["gidx"]), str(meta["scenario"])
            design = {key: float(value) for key, value in zip(design_keys, grid[gidx])}
            jid = pool.submit(
                scenario_name=scenario,
                scenario_cfg=scenario_cfgs[scenario],
                design=design,
                seed=int(meta["seed"]),
                num_episodes=1,
                job_id=f"audit3-{stage_id}-t{int(meta['trial_index']):06d}-{scenario}-s{int(meta['seed'])}",
            )
            active[jid] = {**meta, "submitted_at": time.time()}
            submitted += 1

    try:
        submit_until_full()
        while completed < len(plans):
            terminal = pool.poll()
            if not terminal:
                if getattr(pool, "fatal_error", None) and getattr(pool, "alive_workers", 1) <= 0:
                    raise RuntimeError(f"CARLA worker pool failed: {pool.fatal_error}")
                time.sleep(max(0.0, float(poll_interval_s)))
                continue
            completed_batch: Dict[ResultKey, Dict[str, object]] = {}
            for record in terminal:
                meta = active.pop(record.job_id, None)
                if meta is None:
                    raise RuntimeError(f"pool returned unknown/duplicate job {record.job_id}")
                episode = pool_record_to_episode_result(record, require_min_d=False)
                scenario, gidx = str(meta["scenario"]), int(meta["gidx"])
                actual_seed = verified_simulator_seed(episode, int(meta["seed"]))
                uid = str(epmans[scenario].append_one(
                    gidx, episode,
                    EpisodeMeta(scenario=scenario, split="AUDIT3", provenance="audit3_worker_pool"),
                ))
                safety = getattr(episode, "result", {}).get("safety_m", {}) or {}
                try:
                    min_d = float(safety.get("min_d"))
                except (TypeError, ValueError):
                    min_d = None
                malformed = min_d is None or not math.isfinite(min_d)
                if malformed:
                    min_d, violation = None, True
                else:
                    violation = bool(min_d < float(d_safe))
                result = {
                    "ep_uid": uid, "requested_seed": int(meta["seed"]), "seed": int(actual_seed),
                    "min_d": min_d, "violation": bool(violation), "malformed": malformed,
                    "malformed_reason": "missing_or_non_finite_min_d" if malformed else None,
                    "worker_id": getattr(record, "worker_id", None),
                    "attempts": int(getattr(record, "attempts", 0)),
                    "elapsed_seconds": float(time.time() - float(meta["submitted_at"])),
                    "reused": False, "planned_seed": int(meta["seed"]), "planned_seed_executed": True,
                }
                if uid in used_uids:
                    raise RuntimeError(f"episode manager produced duplicate UID {uid}")
                used_uids.add(uid)
                result_key = (int(meta["trial_index"]), scenario)
                episode_results[result_key] = result
                completed_batch[result_key] = result
                sim_usage[gidx] = sim_usage.get(gidx, 0) + 1
                completed += 1
            if checkpoint_callback and completed_batch:
                checkpoint_callback(completed_batch)
            submit_until_full()
    finally:
        for epman in epmans.values():
            epman.flush()

    trials: List[Dict[str, object]] = []
    k = malformed_episodes = reused_jobs = 0
    for trial in trial_plan:
        ti = int(trial["trial_index"])
        scenarios = {s: episode_results[(ti, s)] for s in scenario_names}
        violation = any(bool(r["violation"]) for r in scenarios.values())
        k += int(violation)
        malformed_episodes += sum(int(bool(r.get("malformed"))) for r in scenarios.values())
        reused_jobs += sum(int(bool(r.get("reused"))) for r in scenarios.values())
        trials.append({
            "stage_id": stage_id, "threshold": float(threshold), "trial_index": ti,
            "gidx": int(trial["gidx"]), "violation": violation,
            "malformed": any(bool(r.get("malformed")) for r in scenarios.values()),
            "scenarios": scenarios,
        })
    n = len(trials)
    cp_upper = float(cp_upper_bound(k, n, float(delta)))
    unsafe_bound = float(min(1.0, cp_upper / max(float(P), 1e-12)))
    return ({
        "algo_version": ALGO_VERSION,
        "stage_id": stage_id,
        "threshold": float(threshold),
        "accepted": int(accepted_count),
        "n": n, "k": k, "delta": float(delta), "P": float(P),
        "alpha_frac": float(alpha_frac), "cp_upper": cp_upper,
        "unsafe_bound": unsafe_bound, "passed": bool(unsafe_bound <= float(alpha_frac)),
        "malformed_episodes": malformed_episodes,
        "cache_reuse": {
            "policy": "matching_design_id", "reused_episode_jobs": reused_jobs,
            "fresh_episode_jobs": len(plans) - reused_jobs,
            "unique_uid_within_stage": True,
        },
        "scheduler": {
            "mode": "bounded_concurrent", "max_in_flight": int(max_in_flight),
            "episode_jobs": len(plans), "submitted_jobs": len(jobs),
            "wall_seconds": float(time.time() - started),
        },
        "trials": trials,
    }, sim_usage)
