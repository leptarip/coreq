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

"""Verify a completed portfolio audit and write its audited-set bundle.

The bundle (audit_result.json, accepted_v3.npy, f_min_v3.npy and README.md)
is the input of Phase 2. The command is offline: it never starts
CARLA and refuses to overwrite an existing bundle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(path for path in _HERE.parents if (path / "source").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduction.phase1.run_audit import preflight  # noqa: E402
from source.design_exploration.commons.design_space import get_space_spec  # noqa: E402
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager  # noqa: E402
from source.design_exploration.commons.scenario_labeling import cp_upper_bound  # noqa: E402


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _stage_by_id(manifest: Mapping[str, object], stage_id: str) -> Dict[str, object]:
    matches = [stage for stage in manifest["stages"] if stage.get("id") == stage_id]
    if len(matches) != 1:
        raise ValueError(f"freeze does not contain exactly one stage {stage_id!r}")
    return dict(matches[0])


def _validate_result(
    *,
    result: Mapping[str, object],
    stage: Mapping[str, object],
    manifest: Mapping[str, object],
    plan: Sequence[Mapping[str, object]],
    accepted: np.ndarray,
) -> Tuple[set[str], Dict[Tuple[int, str], Dict[str, object]]]:
    statistics = manifest["statistics"]
    scenarios = [str(item["name"]) for item in manifest["scenarios"]]
    exact = {
        "stage_id": str(stage["id"]),
        "threshold": float(stage["tau"]),
        "accepted": int(stage["accepted_count"]),
        "n": int(stage["n"]),
        "delta": float(stage["delta"]),
        "P": float(statistics["P"]),
        "alpha_frac": float(statistics["alpha_frac"]),
    }
    mismatches = [key for key, value in exact.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"result differs from frozen stage fields: {mismatches}")
    if not bool(result.get("passed")):
        raise ValueError("selected audit stage did not pass")
    trials = result.get("trials")
    if not isinstance(trials, list) or len(trials) != int(stage["n"]):
        raise ValueError("result trials do not match frozen N")
    if len(plan) != len(trials):
        raise ValueError("sampling plan and result have different lengths")

    accepted_set = {int(value) for value in accepted}
    used_uids: set[str] = set()
    fresh: Dict[Tuple[int, str], Dict[str, object]] = {}
    k = malformed_episodes = 0
    for index, (planned, trial) in enumerate(zip(plan, trials)):
        gidx = int(planned["gidx"])
        if (
            int(planned["trial_index"]) != index
            or int(trial.get("trial_index", -1)) != index
            or int(trial.get("gidx", -1)) != gidx
            or gidx not in accepted_set
            or trial.get("stage_id") != stage["id"]
            or float(trial.get("threshold", math.nan)) != float(stage["tau"])
        ):
            raise ValueError(f"trial {index} differs from its frozen plan or membership")
        scenario_results = trial.get("scenarios")
        if not isinstance(scenario_results, Mapping) or set(scenario_results) != set(scenarios):
            raise ValueError(f"trial {index} has incomplete scenario results")
        trial_violation = False
        trial_malformed = False
        for scenario in scenarios:
            episode = dict(scenario_results[scenario])
            uid = str(episode.get("ep_uid", ""))
            if not uid or uid in used_uids:
                raise ValueError(f"trial {index} has missing or duplicate UID {uid!r}")
            used_uids.add(uid)
            requested = int(episode.get("requested_seed", -1))
            actual = int(episode.get("seed", -1))
            reused = bool(episode.get("reused", False))
            planned_seed = int(planned["seeds"][scenario])
            if requested < 0 or actual != requested:
                raise ValueError(f"trial {index} {scenario} does not preserve its seed")
            if bool(episode.get("planned_seed_executed")) == reused:
                raise ValueError(f"trial {index} {scenario} has inconsistent reuse metadata")
            if not reused and (requested != planned_seed or int(episode["planned_seed"]) != planned_seed):
                raise ValueError(f"trial {index} {scenario} differs from its frozen seed")
            malformed = bool(episode.get("malformed", False))
            violation = bool(episode.get("violation", False))
            raw_min_d = episode.get("min_d")
            try:
                min_d = float(raw_min_d)
            except (TypeError, ValueError):
                min_d = None
            if malformed:
                if min_d is not None and math.isfinite(min_d):
                    raise ValueError(f"malformed episode {uid} has finite min_d")
                if not violation:
                    raise ValueError(f"malformed episode {uid} is not conservative")
            else:
                if min_d is None or not math.isfinite(min_d):
                    raise ValueError(f"episode {uid} has invalid min_d")
                if violation != bool(min_d < float(statistics["d_safe"])):
                    raise ValueError(f"episode {uid} label differs from d_safe")
            trial_violation = trial_violation or violation
            trial_malformed = trial_malformed or malformed
            malformed_episodes += int(malformed)
            if not reused:
                fresh[(index, scenario)] = episode
        if bool(trial.get("violation")) != trial_violation:
            raise ValueError(f"trial {index} portfolio label is inconsistent")
        if bool(trial.get("malformed")) != trial_malformed:
            raise ValueError(f"trial {index} malformed flag is inconsistent")
        k += int(trial_violation)

    if int(result.get("k", -1)) != k:
        raise ValueError("result violation count does not match trials")
    if int(result.get("malformed_episodes", -1)) != malformed_episodes:
        raise ValueError("result malformed count does not match trials")
    cp_upper = float(cp_upper_bound(k, len(trials), float(stage["delta"])))
    unsafe_bound = float(min(1.0, cp_upper / float(statistics["P"])))
    if not math.isclose(float(result["cp_upper"]), cp_upper, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("result CP upper bound does not recompute exactly")
    if not math.isclose(float(result["unsafe_bound"]), unsafe_bound, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("result unsafe bound does not recompute exactly")
    if unsafe_bound > float(statistics["alpha_frac"]):
        raise ValueError("recomputed result does not pass")
    return used_uids, fresh


def _checkpoint_results(path: Path, stage: Mapping[str, object]):
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("checkpoint SQLite integrity check failed")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        expected = {"stage_id": str(stage["id"]), "tau": repr(float(stage["tau"])), "n": str(int(stage["n"]))}
        if metadata != expected:
            raise ValueError("checkpoint belongs to a different audit stage")
        rows = {}
        for trial_index, scenario, raw in connection.execute(
            "SELECT trial_index, scenario, result_json FROM episode_results"
        ):
            rows[(int(trial_index), str(scenario))] = json.loads(raw)
        return rows
    finally:
        connection.close()


def _raw_episode_index(run_dir: Path, manifest: Mapping[str, object]):
    space_spec = get_space_spec(str(manifest["design_space"]))
    grid = space_spec.build_grid()
    index = {}
    for scenario_item in manifest["scenarios"]:
        scenario = str(scenario_item["name"])
        manager = EpisodeManager(
            root_dir=str(run_dir / "episodes"),
            scenario_name=scenario,
            space_spec=space_spec,
            GRID=grid,
        )
        for gidx in manager.list_gidx_with_counts():
            for row in manager.episodes(gidx, consume=False):
                uid = str(row.get("ep_uid", ""))
                if not uid or uid in index:
                    raise ValueError(f"raw episode UID is missing or duplicated: {uid!r}")
                index[uid] = (scenario, int(gidx))
    return index


def finalize(freeze_root: Path, run_dir: Path, out_root: Path) -> Path:
    freeze_root, run_dir, out_root = freeze_root.resolve(), run_dir.resolve(), out_root.resolve()
    manifest = preflight(freeze_root)
    run_state_path = run_dir / "run_state.json"
    run_state = _load_json(run_state_path)
    if (
        run_state.get("status") != "complete"
        or run_state.get("protocol") != manifest["protocol"]
        or run_state.get("freeze_root") != str(freeze_root)
        or run_state.get("familywise_method") != manifest["method"]
        or float(run_state.get("familywise_delta", math.nan)) != float(manifest["global_delta"])
    ):
        raise ValueError("the run does not belong to this freeze or is not complete")
    passed_stages = run_state.get("passed_stages")
    if not isinstance(passed_stages, list) or len(passed_stages) != 1:
        raise ValueError("expected exactly one passed audit stage")
    stage_id = str(passed_stages[0]["stage_id"])
    stage = _stage_by_id(manifest, stage_id)
    stage_state = run_state["stages"].get(stage_id)
    if not isinstance(stage_state, Mapping) or stage_state.get("status") != "complete":
        raise ValueError("passed stage is not complete in run state")
    result_path = run_dir / str(stage_state["result_path"])
    result = _load_json(result_path)
    plan_path = freeze_root / str(stage["plan_path"])
    plan = _load_json(plan_path)
    accepted_path = freeze_root / str(stage["accepted_path"])
    accepted = np.asarray(np.load(str(accepted_path), allow_pickle=False), dtype=np.int64)
    f_min_path = freeze_root / str(manifest["f_min_path"])
    scores = np.asarray(np.load(str(f_min_path), allow_pickle=False), dtype=float)
    expected_accepted = np.flatnonzero(scores >= float(stage["tau"])).astype(np.int64)
    if not np.array_equal(accepted, expected_accepted):
        raise ValueError("accepted artifact differs from f_min >= tau")

    used_uids, fresh = _validate_result(
        result=result, stage=stage, manifest=manifest, plan=plan, accepted=accepted
    )
    checkpoint_path = run_dir / f"checkpoint_{stage_id}.sqlite3"
    checkpoint = _checkpoint_results(checkpoint_path, stage)
    if set(checkpoint) != set(fresh):
        raise ValueError("checkpoint fresh-job keys differ from audited trials")
    for key, episode in fresh.items():
        if checkpoint[key] != episode:
            raise ValueError(f"checkpoint result differs from audited trial {key!r}")
    raw_index = _raw_episode_index(run_dir, manifest)
    plan_by_trial = {int(item["trial_index"]): int(item["gidx"]) for item in plan}
    for (trial_index, scenario), episode in fresh.items():
        expected = (scenario, plan_by_trial[trial_index])
        if raw_index.get(str(episode["ep_uid"])) != expected:
            raise ValueError(f"raw episode does not match audited trial {(trial_index, scenario)!r}")
    if set(raw_index) != used_uids:
        raise ValueError("raw episodes differ from the audited episode UIDs")

    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite audited-set bundle: {out_root}")
    out_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = out_root.parent / f".{out_root.name}.tmp-{uuid.uuid4().hex}"
    tmp_root.mkdir()
    try:
        accepted_copy = tmp_root / "accepted_v3.npy"
        scores_copy = tmp_root / "f_min_v3.npy"
        shutil.copy2(accepted_path, accepted_copy)
        shutil.copy2(f_min_path, scores_copy)
        audit_result = {
            "schema_version": 1,
            "protocol": "portfolio_audit_audited_set",
            "status": "passed",
            "design_space": manifest["design_space"],
            "grid_rows": manifest["grid_rows"],
            "design_keys": manifest["design_keys"],
            "stage_id": stage_id,
            "tau": float(stage["tau"]),
            "audited_count": int(accepted.size),
            "accepted_path": "accepted_v3.npy",
            "f_min_path": "f_min_v3.npy",
            "scenarios": [str(item["name"]) for item in manifest["scenarios"]],
            "statistics": {
                **manifest["statistics"],
                "method": manifest["method"],
                "global_delta": manifest["global_delta"],
                "stage_delta": stage["delta"],
                "n": result["n"],
                "k": result["k"],
                "cp_upper": result["cp_upper"],
                "unsafe_bound": result["unsafe_bound"],
                "malformed_episodes": result["malformed_episodes"],
            },
            "execution": {
                "episode_jobs": result["scheduler"]["episode_jobs"],
                "fresh_episode_jobs": result["cache_reuse"]["fresh_episode_jobs"],
                "reused_episode_jobs": result["cache_reuse"]["reused_episode_jobs"],
                "referenced_episode_uids": len(used_uids),
                "raw_episode_records": len(raw_index),
            },
        }
        _write_json(tmp_root / "audit_result.json", audit_result)
        (tmp_root / "README.md").write_text(
            "# Phase 1 audited set\n\n"
            f"The portfolio audit passed at `tau = {stage['tau']!r}` with `N={result['n']}` paired trials and "
            f"`k={result['k']}` violations. It admits {accepted.size:,} of the {manifest['grid_rows']:,} V3 designs.\n\n"
            "`audit_result.json` records the audit statistics. `f_min_v3.npy` holds the portfolio safety score of every "
            "design and `accepted_v3.npy` the indices of the audited set (`f_min >= tau`).\n",
            encoding="utf-8",
        )
        os.replace(str(tmp_root), str(out_root))
    except BaseException:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return out_root


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    out = finalize(args.freeze_root, args.run_dir, args.out_root)
    audit_result = _load_json(out / "audit_result.json")
    print(json.dumps({
        "status": audit_result["status"],
        "tau": audit_result["tau"],
        "audited_count": audit_result["audited_count"],
        "n": audit_result["statistics"]["n"],
        "k": audit_result["statistics"]["k"],
        "out_root": str(out),
        "simulator_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
