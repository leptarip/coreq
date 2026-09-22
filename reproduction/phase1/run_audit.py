#!/usr/bin/env python3
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

"""Portfolio audit: freeze the declared stages, check them, and execute them.

``freeze`` fixes each stage's candidate set, sample size and seeds before any
episode is run. ``freeze`` and ``preflight`` are offline; CARLA is started only
by ``execute``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(p for p in _HERE.parents if (p / "source").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from source.design_exploration.audit.audit3_portfolio import (  # noqa: E402
    ALGO_VERSION,
    build_frozen_stage_plans,
    build_reuse_assignments,
    run_parallel_cached_portfolio_stage,
    should_run_stage,
    validate_declaration,
)
from source.design_exploration.commons.design_space import get_space_spec  # noqa: E402
from source.design_exploration.commons.scenario_labeling import cp_upper_bound  # noqa: E402

SCHEMA_VERSION = 1

if TYPE_CHECKING:
    import sqlite3


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_json_list(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _resolve(base: Path, value: object) -> Path:
    if value is None:
        raise ValueError("required path is missing")
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _scenario_config(config: Mapping[str, object], base: Path) -> Tuple[List[str], Dict[str, str]]:
    raw = config.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.scenarios must be a non-empty list")
    names: List[str] = []
    paths: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("each scenario must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name in paths:
            raise ValueError(f"scenario names must be non-empty and unique: {name!r}")
        cfg_path = _resolve(base, item.get("config"))
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        names.append(name)
        paths[name] = str(cfg_path)
    return names, paths


def _max_passing_failures(n: int, delta: float, P: float, alpha_frac: float) -> int:
    """Largest k whose one-sided CP bound meets the declared audit target."""
    target = float(P) * float(alpha_frac)
    low, high, answer = 0, int(n), -1
    while low <= high:
        middle = (low + high) // 2
        if float(cp_upper_bound(middle, int(n), float(delta))) <= target:
            answer, low = middle, middle + 1
        else:
            high = middle - 1
    return answer


def _load_scores(
    surrogate: Mapping[str, object], *, base: Path, scenario_names: Sequence[str], grid: np.ndarray
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load the f_min score vector from arrays or explicit model metas."""
    mode = str(surrogate.get("mode", ""))
    detail: Dict[str, object] = {"mode": mode}
    if mode == "precomputed_f_min":
        path = _resolve(base, surrogate.get("path"))
        scores = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        detail["path"] = str(path)
    elif mode == "precomputed_p_safe":
        per = surrogate.get("per_scenario")
        if not isinstance(per, Mapping) or set(per) != set(scenario_names):
            raise ValueError("surrogate.per_scenario must exactly match configured scenarios")
        arrays, resolved = [], {}
        for scenario in scenario_names:
            path = _resolve(base, per[scenario])
            array = np.asarray(np.load(path, allow_pickle=False), dtype=float)
            if array.shape != (len(grid),) or not np.isfinite(array).all():
                raise ValueError(f"invalid p_safe array for {scenario}: {array.shape}")
            arrays.append(array)
            resolved[scenario] = str(path)
        scores = np.min(np.vstack(arrays), axis=0)
        detail["per_scenario"] = resolved
    elif mode == "model_meta":
        per = surrogate.get("per_scenario")
        if not isinstance(per, Mapping) or set(per) != set(scenario_names):
            raise ValueError("surrogate.per_scenario must exactly match configured scenarios")
        resolved = {name: str(_resolve(base, per[name])) for name in scenario_names}
        for name, path in resolved.items():
            meta = _load_json(Path(path))
            model_value = meta.get("model_path")
            if model_value:
                model_path = Path(str(model_value)).expanduser()
                if not model_path.is_absolute():
                    model_path = Path(path).parent / model_path
                model_path = model_path.resolve()
            else:
                stem = Path(path).stem
                prefix = stem[:-5] if stem.endswith("_meta") else stem
                candidates = sorted(
                    p for p in Path(path).parent.iterdir()
                    if p.is_file() and p.name.startswith(prefix)
                    and p.suffix in {".joblib", ".pkl", ".pickle"}
                )
                if not candidates:
                    raise ValueError(f"cannot infer model artifact from {path}")
                model_path = candidates[0].resolve()
        from source.design_exploration.commons.surrogate_loading import load_surrogate_family_from_model_metas
        predictor = load_surrogate_family_from_model_metas(
            resolved, model_family=surrogate.get("model_family"),
            df_override=surrogate.get("df_override"),
        )
        p_safe = [
            1.0 - np.asarray(predictor.per_scenario[name](grid), dtype=float)
            for name in scenario_names
        ]
        if any(array.shape != (len(grid),) or not np.isfinite(array).all() for array in p_safe):
            raise ValueError("model-meta prediction produced invalid p_safe arrays")
        scores = np.min(np.vstack(p_safe), axis=0)
        detail.update(per_scenario=resolved, model_family=predictor.name)
    else:
        raise ValueError(
            "surrogate.mode must be precomputed_f_min, precomputed_p_safe, or model_meta"
        )
    if scores.shape != (len(grid),):
        raise ValueError(f"score shape {scores.shape} does not match grid shape ({len(grid)},)")
    if not np.isfinite(scores).all():
        raise ValueError("score vector contains non-finite values")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("safety scores must lie in [0, 1]")
    return scores, detail


def freeze(config_path: Path, freeze_root_override: Optional[Path] = None) -> Path:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}")
    base = config_path.parent
    declaration_path = _resolve(base, config.get("declaration_path"))
    declaration = _load_json(declaration_path)
    scenario_names, scenario_paths = _scenario_config(config, base)
    stages = validate_declaration(declaration, scenario_names=scenario_names)

    space_name = str(config.get("design_space", ""))
    space_spec = get_space_spec(space_name)
    grid = np.asarray(space_spec.build_grid(), dtype=float)
    if grid.shape[0] == 0 or np.unique(grid, axis=0).shape[0] != grid.shape[0]:
        raise ValueError(f"design space {space_name!r} is empty or not one-to-one")
    surrogate = config.get("surrogate")
    if not isinstance(surrogate, Mapping):
        raise ValueError("config.surrogate must be an object")
    scores, surrogate_detail = _load_scores(
        surrogate, base=base, scenario_names=scenario_names, grid=grid
    )
    stage_summaries, plans = build_frozen_stage_plans(
        scores=scores, stages=stages, scenario_names=scenario_names
    )

    statistics = config.get("statistics") or {}
    if not isinstance(statistics, Mapping):
        raise ValueError("config.statistics must be an object")
    P = float(statistics.get("P", 0.05))
    alpha_frac = float(statistics.get("alpha_frac", 0.05))
    d_safe = float(statistics.get("d_safe", 0.5))
    if not (0 < P <= 1 and 0 < alpha_frac <= 1 and math.isfinite(d_safe)):
        raise ValueError("invalid P, alpha_frac, or d_safe")
    for stage in stage_summaries:
        max_failures = _max_passing_failures(
            int(stage["n"]), float(stage["delta"]), P, alpha_frac
        )
        if max_failures < 0:
            raise ValueError(
                f"stage {stage['id']} can never pass even with k=0; increase n or delta"
            )
        stage["max_passing_failures"] = max_failures
    paths = config.get("paths") or {}
    if not isinstance(paths, Mapping):
        raise ValueError("config.paths must be an object")
    configured_root = paths.get("freeze_root")
    if freeze_root_override is None and not configured_root:
        raise ValueError("set paths.freeze_root or pass --freeze-root")
    freeze_root = (
        freeze_root_override.resolve() if freeze_root_override is not None
        else _resolve(base, configured_root)
    )
    if freeze_root.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze root: {freeze_root}")
    freeze_root.mkdir(parents=True)
    shutil.copy2(config_path, freeze_root / "config.json")
    shutil.copy2(declaration_path, freeze_root / "declaration.json")
    np.save(freeze_root / "f_min.npy", scores, allow_pickle=False)
    stage_records: List[Dict[str, object]] = []
    for stage in stage_summaries:
        stage_id = str(stage["id"])
        stage_copy = dict(stage)
        accepted = np.asarray(stage_copy.pop("accepted_idx"), dtype=np.int64)
        accepted_path = freeze_root / f"accepted_{stage_id}.npy"
        plan_path = freeze_root / f"plan_{stage_id}.json"
        np.save(accepted_path, accepted, allow_pickle=False)
        _write_json_atomic(plan_path, plans[stage_id])
        stage_records.append({
            **stage_copy,
            "accepted_path": accepted_path.name,
            "plan_path": plan_path.name,
        })
    runtime = config.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        raise ValueError("config.runtime must be an object")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "portfolio_audit_freeze", "status": "frozen",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "freeze_root": str(freeze_root), "method": "bonferroni",
        "global_delta": float(declaration["global_delta"]),
        "allocated_delta": float(sum(float(s["delta"]) for s in stages)),
        "cache_reuse": "matching_design_id", "design_space": space_name,
        "design_keys": list(space_spec.keys), "grid_rows": int(len(grid)),
        "scenarios": [{"name": n, "config": scenario_paths[n]} for n in scenario_names],
        "surrogate": surrogate_detail, "f_min_path": "f_min.npy",
        "statistics": {"P": P, "alpha_frac": alpha_frac, "d_safe": d_safe},
        "stages": stage_records,
        "runtime": dict(runtime),
        "configured_execution_root": (
            str(_resolve(base, paths["execution_root"])) if paths.get("execution_root") else None
        ),
        "algo_version": ALGO_VERSION,
    }
    _write_json_atomic(freeze_root / "manifest.json", manifest)
    return freeze_root


def preflight(freeze_root: Path) -> Dict[str, object]:
    """Check a freeze offline: candidate sets, plan sizes and stage feasibility."""
    freeze_root = freeze_root.resolve()
    manifest = _load_json(freeze_root / "manifest.json")
    if manifest.get("protocol") != "portfolio_audit_freeze":
        raise ValueError("not a portfolio-audit freeze")
    spec = get_space_spec(str(manifest["design_space"]))
    if list(spec.keys) != manifest["design_keys"]:
        raise ValueError("design-space keys differ from the freeze")
    scores = np.asarray(np.load(freeze_root / str(manifest["f_min_path"]), allow_pickle=False))
    statistics = manifest["statistics"]
    for stage in manifest["stages"]:
        expected = np.flatnonzero(scores >= float(stage["tau"])).astype(np.int64)
        saved = np.load(freeze_root / str(stage["accepted_path"]), allow_pickle=False)
        if not np.array_equal(saved, expected):
            raise ValueError(f"candidate set differs from f_min >= tau for stage {stage['id']}")
        plan = _load_json_list(freeze_root / str(stage["plan_path"]))
        if len(plan) != int(stage["n"]):
            raise ValueError(f"plan length mismatch for stage {stage['id']}")
        max_failures = _max_passing_failures(
            int(stage["n"]), float(stage["delta"]),
            float(statistics["P"]), float(statistics["alpha_frac"]),
        )
        if max_failures < 0 or int(stage["max_passing_failures"]) != max_failures:
            raise ValueError(f"stage {stage['id']} can never pass or has a wrong failure limit")
    return manifest


def _open_checkpoint(
    path: Path, *, stage_id: str, tau: float, n: int
) -> Tuple["sqlite3.Connection", Dict[Tuple[int, str], Dict[str, object]]]:
    """Open an atomic append-only stage checkpoint and load completed jobs."""
    import sqlite3

    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS episode_results ("
        "trial_index INTEGER NOT NULL, scenario TEXT NOT NULL, result_json TEXT NOT NULL, "
        "PRIMARY KEY (trial_index, scenario))"
    )
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    expected = {"stage_id": stage_id, "tau": repr(float(tau)), "n": str(int(n))}
    if metadata and metadata != expected:
        connection.close()
        raise ValueError(f"checkpoint identity mismatch for stage {stage_id}")
    if not metadata:
        with connection:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
            )
    results: Dict[Tuple[int, str], Dict[str, object]] = {}
    for trial_index, scenario, raw in connection.execute(
        "SELECT trial_index, scenario, result_json FROM episode_results"
    ):
        value = json.loads(raw)
        if not isinstance(value, dict):
            connection.close()
            raise ValueError("checkpoint contains a non-object episode result")
        results[(int(trial_index), str(scenario))] = value
    return connection, results


def _checkpoint_new_results(
    connection: "sqlite3.Connection",
    known: set[Tuple[int, str]],
    results: Mapping[Tuple[int, str], Mapping[str, object]],
) -> None:
    rows = [
        (key[0], key[1], json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
        for key, value in results.items() if key not in known
    ]
    if not rows:
        return
    with connection:
        connection.executemany(
            "INSERT INTO episode_results(trial_index, scenario, result_json) VALUES (?, ?, ?)",
            rows,
        )
    known.update((int(row[0]), str(row[1])) for row in rows)


def _raw_uid_index(epmans) -> Dict[str, Tuple[str, int]]:
    index: Dict[str, Tuple[str, int]] = {}
    for scenario, manager in epmans.items():
        for gidx in manager.list_gidx_with_counts():
            for row in manager.episodes(gidx, consume=False):
                uid = str(row.get("ep_uid", ""))
                if not uid or uid in index:
                    raise ValueError(f"raw episode UID is missing or duplicated: {uid!r}")
                index[uid] = (str(scenario), int(gidx))
    return index


def _validate_checkpoint_raw(
    results: Mapping[Tuple[int, str], Mapping[str, object]],
    plan: Sequence[Mapping[str, object]],
    epmans,
) -> None:
    raw = _raw_uid_index(epmans)
    by_trial = {int(item["trial_index"]): int(item["gidx"]) for item in plan}
    for (trial_index, scenario), result in results.items():
        expected = (scenario, by_trial.get(trial_index, -1))
        if raw.get(str(result.get("ep_uid", ""))) != expected:
            raise ValueError(
                f"checkpoint episode has no matching durable raw record: {(trial_index, scenario)!r}"
            )


def _load_resume_state(run_dir: Path, freeze_root: Path) -> Dict[str, object]:
    state = _load_json(run_dir / "run_state.json")
    if state.get("status") == "complete":
        raise ValueError("run is already complete")
    if state.get("freeze_root") != str(freeze_root.resolve()):
        raise ValueError("resume run records a different freeze root")
    if not isinstance(state.get("stages"), dict):
        raise ValueError("resume run has malformed stage state")
    state["status"] = "running"
    state.pop("error", None)
    return state


def execute(freeze_root: Path, execution_root: Optional[Path], resume: Optional[Path]) -> Path:
    manifest = preflight(freeze_root)
    runtime = manifest.get("runtime") or {}
    carla_path = Path(str(runtime.get("carla_path", ""))).resolve()
    agents_path = Path(str(runtime.get("carla_agents_path", ""))).resolve()
    python_executable = Path(str(runtime.get("python_executable") or sys.executable)).resolve()
    if not carla_path.is_file() or not (agents_path / "agents").is_dir():
        raise ValueError("runtime CARLA executable or agents directory is missing")
    if Path(sys.executable).resolve() != python_executable:
        raise ValueError(f"execute with frozen python_executable {python_executable}")
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))

    # Simulator-bearing imports occur only after the explicit execute command.
    import source.simulation_environment.cfg.cfg_parser as cfg_parser
    from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
    from source.design_exploration.simulator_interface.carla_worker_pool import CarlaWorkerPool

    configured = manifest.get("configured_execution_root")
    root_value = execution_root or (Path(str(configured)) if configured else None)
    if resume is None and root_value is None:
        raise ValueError("set paths.execution_root or pass --execution-root")
    if resume is None:
        run_dir = root_value.resolve() / (
            f"audit_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "protocol": manifest["protocol"], "status": "running",
            "freeze_root": str(freeze_root.resolve()), "stages": {},
        }
    else:
        run_dir = resume.resolve()
        state = _load_resume_state(run_dir, freeze_root)
    _write_json_atomic(run_dir / "run_state.json", state)

    scenario_names = [str(x["name"]) for x in manifest["scenarios"]]
    scenario_cfgs = {
        str(x["name"]): cfg_parser.parse(str(x["config"])) for x in manifest["scenarios"]
    }
    spec = get_space_spec(str(manifest["design_space"]))
    grid = spec.build_grid()
    epmans = {
        name: EpisodeManager(
            root_dir=str(run_dir / "episodes"), scenario_name=name, space_spec=spec, GRID=grid
        ) for name in scenario_names
    }
    statistics = manifest["statistics"]
    num_workers = int(runtime.get("num_workers", 3))
    max_in_flight = int(runtime.get("max_in_flight", 2 * num_workers))
    if num_workers <= 0 or max_in_flight <= 0:
        raise ValueError("num_workers and max_in_flight must be positive")

    outcomes: Dict[str, Dict[str, object]] = {}
    # Only unconditional stages can supply reusable observations. Their
    # availability is fixed independently of all observed outcomes.
    reusable_trials: List[Dict[str, object]] = []
    pool = None
    try:
        for stage in manifest["stages"]:
            stage_id = str(stage["id"])
            prior_state = (state.get("stages") or {}).get(stage_id)
            if isinstance(prior_state, Mapping) and prior_state.get("status") == "complete":
                saved_result_path = run_dir / str(prior_state["result_path"])
                result = _load_json(saved_result_path)
                if (
                    result.get("stage_id") != stage_id
                    or float(result.get("threshold", math.nan)) != float(stage["tau"])
                    or int(result.get("n", -1)) != int(stage["n"])
                    or float(result.get("delta", math.nan)) != float(stage["delta"])
                ):
                    raise ValueError(f"completed result does not match frozen stage {stage_id}")
                outcomes[stage_id] = result
                if stage.get("run_if") == "always":
                    reusable_trials.extend(result.get("trials", []))
                continue
            if not should_run_stage(stage, outcomes):
                state["stages"][stage_id] = {"status": "skipped", "reason": "predeclared_gate_false"}
                outcomes[stage_id] = {"status": "skipped", "passed": False}
                _write_json_atomic(run_dir / "run_state.json", state)
                continue

            plan = _load_json_list(freeze_root / str(stage["plan_path"]))
            reused = build_reuse_assignments(
                trial_plan=plan, scenario_names=scenario_names,
                prior_trials=reusable_trials,
            )
            checkpoint_path = run_dir / f"checkpoint_{stage_id}.sqlite3"
            checkpoint_db, resumed = _open_checkpoint(
                checkpoint_path, stage_id=stage_id, tau=float(stage["tau"]), n=int(stage["n"])
            )
            try:
                _validate_checkpoint_raw(resumed, plan, epmans)
                expected_jobs = len(plan) * len(scenario_names)
                if len(set(reused).union(resumed)) < expected_jobs and pool is None:
                    pool = CarlaWorkerPool(
                        num_workers=num_workers, base_port=int(runtime.get("base_port", 2000)),
                        carla_path=str(carla_path), output_root=str(run_dir / "sim_result"),
                        algo_version=ALGO_VERSION,
                        heartbeat_s=float(runtime.get("heartbeat_s", 5.0)),
                        max_retries=int(runtime.get("max_retries", 10)),
                        spawn_stagger_s=float(runtime.get("spawn_stagger_s", 6.0)),
                        max_worker_respawns=int(runtime.get("max_worker_respawns", 2)),
                    )
                state["stages"][stage_id] = {
                    "status": "running", "checkpoint_path": checkpoint_path.name
                }
                _write_json_atomic(run_dir / "run_state.json", state)
            except BaseException:
                checkpoint_db.close()
                raise

            checkpointed_keys = set(resumed)

            def checkpoint_callback(results) -> None:
                for manager in epmans.values():
                    manager.flush()
                _checkpoint_new_results(checkpoint_db, checkpointed_keys, results)

            try:
                result, sim_usage = run_parallel_cached_portfolio_stage(
                    stage_id=stage_id, threshold=float(stage["tau"]),
                    accepted_count=int(stage["accepted_count"]), trial_plan=plan,
                    scenario_names=scenario_names, scenario_cfgs=scenario_cfgs,
                    grid=grid, design_keys=spec.keys, pool=pool, epmans=epmans,
                    P=float(statistics["P"]), alpha_frac=float(statistics["alpha_frac"]),
                    delta=float(stage["delta"]), d_safe=float(statistics["d_safe"]),
                    max_in_flight=max_in_flight,
                    reused_episode_results=reused, resumed_episode_results=resumed,
                    checkpoint_callback=checkpoint_callback,
                )
            finally:
                checkpoint_db.close()
            result_path = Path(f"result_{stage_id}.json")
            _write_json_atomic(run_dir / result_path, result)
            _write_json_atomic(
                run_dir / f"sim_usage_{stage_id}.json", {str(k): v for k, v in sim_usage.items()}
            )
            state["stages"][stage_id] = {
                "status": "complete", "passed": bool(result["passed"]),
                "result_path": str(result_path),
            }
            _write_json_atomic(run_dir / "run_state.json", state)
            outcomes[stage_id] = result
            if stage.get("run_if") == "always":
                reusable_trials.extend(result["trials"])
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json_atomic(run_dir / "run_state.json", state)
        raise
    finally:
        if pool is not None:
            pool.shutdown()
        for manager in epmans.values():
            manager.flush()

    passed = [
        {"stage_id": stage_id, "threshold": result["threshold"]}
        for stage_id, result in outcomes.items() if result.get("passed")
    ]
    state.update(
        status="complete", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        familywise_method="bonferroni", familywise_delta=float(manifest["global_delta"]),
        passed_stages=passed,
    )
    state.pop("error", None)
    _write_json_atomic(run_dir / "run_state.json", state)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio audit with stages declared before sampling")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_cmd = commands.add_parser("freeze", help="Freeze inputs, memberships, plans, and seeds offline")
    freeze_cmd.add_argument("--config", type=Path, required=True)
    freeze_cmd.add_argument("--freeze-root", type=Path)
    check_cmd = commands.add_parser("preflight", help="Verify every frozen artifact offline")
    check_cmd.add_argument("--freeze-root", type=Path, required=True)
    execute_cmd = commands.add_parser("execute", help="Explicitly start workers and run the frozen audit")
    execute_cmd.add_argument("--freeze-root", type=Path, required=True)
    execute_cmd.add_argument("--execution-root", type=Path)
    execute_cmd.add_argument("--resume", type=Path)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        root = freeze(args.config, args.freeze_root)
        print(f"Audit frozen at {root}. CARLA was not started.")
    elif args.command == "preflight":
        manifest = preflight(args.freeze_root)
        print(json.dumps({
            "status": "preflight_passed", "freeze_root": manifest["freeze_root"],
            "design_space": manifest["design_space"],
            "stages": [
                {
                    k: s[k]
                    for k in (
                        "id", "tau", "delta", "n", "max_passing_failures",
                        "accepted_count",
                    )
                }
                for s in manifest["stages"]
            ],
        }, indent=2))
        print("CARLA was not started.")
    else:
        run_dir = execute(args.freeze_root, args.execution_root, args.resume)
        print(f"Audit complete: {run_dir}")


if __name__ == "__main__":
    main()
