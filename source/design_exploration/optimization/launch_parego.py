#!/usr/bin/env python3
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
Phase 2: ParEGO over the audited set.

Input is the audited-set bundle written in Phase 1. The launcher
1. validates the bundle and freezes the candidate set (the audited designs) and
   the run settings into ``<run_dir>/run_manifest.json``;
2. re-executes this module under Python 3.11, which runs the ParEGO client;
3. the client starts the Python 3.8 simulator server (``sim_ipc_server.py``).

The optimizer policy is fixed (``parego_qrf_common.PROFILE``). Seeds, budget,
cost profile and warm-start episode caches are options; running the launcher
with several ``--random-seed`` / ``--episode-seed-offset`` pairs gives a
multi-start search.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from source.design_exploration.commons import design_space
from source.design_exploration.commons.python_paths import (
    get_python311_path,
    get_python38_path,
    require_python311_path,
    require_python38_path,
)
from source.design_exploration.commons.scenario_defaults import SCENARIO_1, SCENARIO_2
from source.design_exploration.optimization.cost_model import (
    build_cost_model,
    cost_model_manifest,
    resolve_hypervolume_reference,
)
from source.design_exploration.optimization.parego_qrf_common import (
    DEFAULT_RHO,
    PROFILE,
    PROFILE_SPEC,
)


_HERE = Path(__file__).resolve()
_REPO_ROOT = next(cand for cand in _HERE.parents if (cand / "source").is_dir())
CHILD_ENV = "PAREGO_CHILD_MANIFEST"
AUDIT_PROTOCOL = "portfolio_audit_audited_set"

SCENARIO_BY_NAME = {
    str(SCENARIO_1["name"]): str(SCENARIO_1["config"]),
    str(SCENARIO_2["name"]): str(SCENARIO_2["config"]),
}

# Objective and QRF settings are part of the optimizer policy, not options.
POLICY = {
    "profile": PROFILE,
    "perf_metric": "kpi.vel_at_min_d",
    "scenario_agg": "mean",
    "portfolio_agg": "mean",
    "maximize_metric": True,
    "rf_quantile": 0.25,
    "rf_estimators": 200,
    "rf_min_samples_leaf": 3,
    "rho": DEFAULT_RHO,
}

DEFAULTS = {
    "audited_set": str(_REPO_ROOT / "results" / "phase1" / "audited_set"),
    "cost_profile_config": str(_REPO_ROOT / "configuration" / "cost_profiles.json"),
    "cost_profile_id": "central",
    "out_dir": str(_REPO_ROOT / "output" / "phase2"),
    "budget": 30,
    "episodes_per_scenario": 20,
    "random_seed": 11,
    "episode_seed_offset": 400_000_000,
    "num_carla_instances": 3,
    "episode_chunk_size": 1,
    "max_in_flight": 6,
    "operational_gate_window_designs": 5,
    "max_seconds_per_paired_trial": 12.0,
    "max_retries_per_100_episodes": 10.0,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ParEGO over the audited set written by the Phase 1 audit."
    )
    p.add_argument("--audited-set", default=DEFAULTS["audited_set"],
                   help="Audited-set bundle directory (audit_result.json, manifest.json, ...).")
    p.add_argument("--resume-run-dir", default=None,
                   help="Continue an existing run after deterministic replay of its history.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Validate the bundle and write the run manifest without starting CARLA.")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"], help="Parent directory for new runs.")
    p.add_argument("--run-dir", default=None, help="Exact, not yet existing directory for a new run.")
    p.add_argument("--python311", default=None, help="Python 3.11 interpreter (else $PYTHON311).")
    p.add_argument("--python38", default=None, help="Python 3.8 interpreter (else $PYTHON38).")
    p.add_argument("--carla-path", default=os.environ.get("CARLA_PATH", ""),
                   help="CARLA server executable (else $CARLA_PATH).")
    p.add_argument("--carla-agents-path", default=os.environ.get("CARLA_AGENTS_PATH", ""),
                   help="Directory containing CARLA's agents package (else $CARLA_AGENTS_PATH).")
    p.add_argument("--warm-start-episode-root", action="append", dest="warm_start_episode_roots",
                   default=[], help="Episode cache to reuse; repeat for several roots.")
    p.add_argument("--cost-profile-config", default=DEFAULTS["cost_profile_config"])
    p.add_argument("--cost-profile-id", default=DEFAULTS["cost_profile_id"])
    for key in ("budget", "episodes_per_scenario", "random_seed", "episode_seed_offset",
                "num_carla_instances", "episode_chunk_size", "max_in_flight",
                "operational_gate_window_designs"):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULTS[key])
    for key in ("max_seconds_per_paired_trial", "max_retries_per_100_episodes"):
        p.add_argument("--" + key.replace("_", "-"), type=float, default=DEFAULTS[key])
    return p


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def load_audited_set(bundle: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Read an audited-set folder and return its audit result and audited design indices."""
    bundle = bundle.resolve()
    result_path = bundle / "audit_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"No audit_result.json in {bundle}")
    audit_result = _load_json(result_path)
    if audit_result.get("protocol") != AUDIT_PROTOCOL or audit_result.get("status") != "passed":
        raise RuntimeError("The audit did not pass.")
    space_spec = design_space.get_space_spec(audit_result["design_space"])
    grid_rows = space_spec.build_grid().shape[0]
    f_min = np.asarray(np.load(bundle / audit_result["f_min_path"]), dtype=float).reshape(-1)
    if f_min.shape[0] != grid_rows:
        raise RuntimeError("The audit scores do not cover the design grid.")
    candidate_indices = np.flatnonzero(f_min >= float(audit_result["tau"])).astype(int)
    accepted = np.asarray(np.load(bundle / audit_result["accepted_path"]), dtype=int).reshape(-1)
    if not np.array_equal(accepted, candidate_indices):
        raise RuntimeError("The audited set is not the set of designs with f_min >= tau.")
    unknown = [name for name in audit_result["scenarios"] if name not in SCENARIO_BY_NAME]
    if unknown:
        raise RuntimeError(f"Audited scenarios are not defined in scenario_defaults.py: {unknown}")
    return audit_result, candidate_indices


def _resolve_interpreters(cfg: Dict[str, Any]) -> None:
    cfg["python311"] = require_python311_path(
        cfg.get("python311") or get_python311_path(),
        cli_flag="--python311", purpose="the ParEGO optimizer client",
    )
    cfg["python38"] = require_python38_path(
        cfg.get("python38") or get_python38_path(),
        cli_flag="--python38", purpose="the ParEGO simulator server",
    )
    if cfg.get("prepare_only"):
        return
    carla_path = Path(str(cfg.get("carla_path") or "")).resolve()
    if not carla_path.is_file() or not os.access(str(carla_path), os.X_OK):
        raise FileNotFoundError(f"A CARLA executable is required (--carla-path): {carla_path}")
    agents_path = Path(str(cfg.get("carla_agents_path") or "")).resolve()
    if not (agents_path / "agents").is_dir():
        raise FileNotFoundError(
            f"CARLA's agents package was not found (--carla-agents-path): {agents_path}"
        )
    cfg["carla_path"] = str(carla_path)
    cfg["carla_agents_path"] = str(agents_path)


def _launch_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "python38": str(cfg["python38"]),
        "carla_path": str(cfg.get("carla_path") or ""),
        "carla_agents_path": str(cfg.get("carla_agents_path") or ""),
        "budget": int(cfg["budget"]),
        "num_carla_instances": int(cfg["num_carla_instances"]),
        "episode_chunk_size": int(cfg["episode_chunk_size"]),
        "max_in_flight": int(cfg["max_in_flight"]),
        "operational_gate_window_designs": int(cfg["operational_gate_window_designs"]),
        "max_seconds_per_paired_trial": cfg["max_seconds_per_paired_trial"],
        "max_retries_per_100_episodes": cfg["max_retries_per_100_episodes"],
    }


def build_run_manifest(cfg: Dict[str, Any]) -> Path:
    """Freeze a new run: candidate set, cost model, hypervolume reference and settings."""
    bundle = Path(str(cfg["audited_set"])).resolve()
    audit_result, candidate_indices = load_audited_set(bundle)
    space_spec = design_space.get_space_spec(audit_result["design_space"])
    cost_config = Path(str(cfg["cost_profile_config"])).resolve()
    cost_model = build_cost_model(space_spec, config_path=str(cost_config),
                                  profile_id=str(cfg["cost_profile_id"]))
    hv_reference = resolve_hypervolume_reference(cost_model, space_spec, candidate_indices)

    if cfg.get("run_dir"):
        run_dir = Path(str(cfg["run_dir"])).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_root = Path(str(cfg["out_dir"])).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        stem = f"parego_{PROFILE}_seed{int(cfg['random_seed'])}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = out_root / stem
        counter = 1
        while run_dir.exists():
            run_dir = out_root / f"{stem}_{counter:02d}"
            counter += 1
        run_dir.mkdir(parents=True)
    candidate_indices_path = run_dir / "candidate_indices.npy"
    np.save(candidate_indices_path, candidate_indices)

    manifest = {
        "audit_result": {
            "path": str(bundle / "audit_result.json"),
            "tau": float(audit_result["tau"]),
            "statistics": audit_result["statistics"],
        },
        "space_spec_name": space_spec.name,
        "candidate_count": int(candidate_indices.size),
        "candidate_indices_path": str(candidate_indices_path),
        "scenario_specs": [
            {
                "name": str(name),
                "cfg_path": SCENARIO_BY_NAME[str(name)],
                "episodes_root": str(run_dir / "episodes" / str(name)),
            }
            for name in audit_result["scenarios"]
        ],
        "optimization": {
            **POLICY,
            "resolved_policy": dataclasses.asdict(PROFILE_SPEC),
            "out_dir": str(run_dir),
            "warm_start_episode_roots": [str(p) for p in cfg.get("warm_start_episode_roots") or []],
            "episodes_per_scenario": int(cfg["episodes_per_scenario"]),
            "random_seed": int(cfg["random_seed"]),
            "episode_seed_offset": int(cfg["episode_seed_offset"]),
            "cost_model": cost_model_manifest(cost_model, space_spec),
            "hv_ref_point": hv_reference["resolved_point"],
            "hypervolume_reference": hv_reference,
            **_launch_settings(cfg),
            "resume": False,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def build_resume_manifest(cfg: Dict[str, Any]) -> Path:
    """Write a launch record for continuing a run; the original manifest is unchanged."""
    run_dir = Path(str(cfg["resume_run_dir"])).resolve()
    source_path = run_dir / "run_manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Resume directory has no run_manifest.json: {run_dir}")
    manifest = _load_json(source_path)
    opt = dict(manifest["optimization"])
    if Path(str(opt["out_dir"])).resolve() != run_dir:
        raise RuntimeError("Resume manifest does not identify its run directory.")
    if opt.get("profile") != PROFILE:
        raise RuntimeError(f"Resume manifest was written for profile {opt.get('profile')!r}.")
    opt.update(_launch_settings(cfg))
    opt["resume"] = True
    manifest["optimization"] = opt
    manifest["continuation_launch"] = {"requested_budget": int(cfg["budget"]), "created_unix": time.time()}
    resume_path = run_dir / "resume_manifest.json"
    _write_json_atomic(resume_path, manifest)
    return resume_path


def run_child_from_manifest(manifest_path: Path) -> None:
    """Python 3.11 side: run the ParEGO client on the run described by the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from source.design_exploration.optimization.parego_qrf_ipc_client import (
        IPCScenarioSpec,
        run_parego_over_candidates,
    )

    manifest = _load_json(manifest_path)
    opt = manifest["optimization"]
    space_spec = design_space.get_space_spec(manifest["space_spec_name"])
    candidate_indices = np.asarray(np.load(manifest["candidate_indices_path"]), dtype=int)
    cost = opt["cost_model"]
    scenario_specs = [
        IPCScenarioSpec(name=sc["name"], cfg_path=sc["cfg_path"], episodes_root=sc["episodes_root"])
        for sc in manifest["scenario_specs"]
    ]
    run_parego_over_candidates(
        space_spec=space_spec,
        scenario_specs=scenario_specs,
        python38=str(opt["python38"]),
        carla_path=str(opt["carla_path"]),
        warm_start_episode_roots=list(opt["warm_start_episode_roots"]),
        candidate_indices=candidate_indices,
        out_dir=str(opt["out_dir"]),
        budget=int(opt["budget"]),
        audited_count=int(manifest["candidate_count"]),
        perf_metric=str(opt["perf_metric"]),
        episodes_per_scenario=int(opt["episodes_per_scenario"]),
        scenario_agg=str(opt["scenario_agg"]),
        portfolio_agg=str(opt["portfolio_agg"]),
        maximize_metric=bool(opt["maximize_metric"]),
        random_seed=int(opt["random_seed"]),
        rf_quantile=float(opt["rf_quantile"]),
        rf_estimators=int(opt["rf_estimators"]),
        rf_min_samples_leaf=int(opt["rf_min_samples_leaf"]),
        rho=float(opt["rho"]),
        profile=str(opt["profile"]),
        num_carla_instances=int(opt["num_carla_instances"]),
        cost_profile_config=str(cost["config_path"]),
        cost_profile_id=str(cost["profile_id"]),
        hv_ref_point=opt["hv_ref_point"],
        hv_reference_metadata=opt["hypervolume_reference"],
        episode_chunk_size=int(opt["episode_chunk_size"]),
        max_in_flight=int(opt["max_in_flight"]),
        episode_seed_offset=int(opt["episode_seed_offset"]),
        operational_gate_window_designs=int(opt["operational_gate_window_designs"]),
        max_seconds_per_paired_trial=opt["max_seconds_per_paired_trial"],
        max_retries_per_100_episodes=opt["max_retries_per_100_episodes"],
        resume=bool(opt["resume"]),
    )


def _stop_optimizer_child(
    proc: subprocess.Popen,
    *,
    interrupt_timeout: float = 20.0,
    terminate_timeout: float = 10.0,
) -> None:
    """Stop the optimizer process group while giving its cleanup handlers time to run."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    def _send(sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    for sig, timeout in (
        (signal.SIGINT, interrupt_timeout),
        (signal.SIGTERM, terminate_timeout),
        (signal.SIGKILL, 5.0),
    ):
        if proc.poll() is not None:
            return
        _send(sig)
        try:
            proc.wait(timeout=timeout)
            return
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            continue


def _run_optimizer_child(cmd: List[str], *, env: Dict[str, str]) -> None:
    """Run the Python 3.11 optimizer in an owned process group."""
    proc = subprocess.Popen(cmd, env=env, cwd=str(_REPO_ROOT), start_new_session=True)
    try:
        return_code = proc.wait()
    except BaseException:
        _stop_optimizer_child(proc)
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def main(argv: Optional[List[str]] = None) -> None:
    child_manifest = os.environ.get(CHILD_ENV)
    if child_manifest:
        run_child_from_manifest(Path(child_manifest))
        return

    cfg = vars(_build_arg_parser().parse_args(argv))
    _resolve_interpreters(cfg)
    manifest_path = build_resume_manifest(cfg) if cfg["resume_run_dir"] else build_run_manifest(cfg)
    if cfg["prepare_only"]:
        print(f"[launch_parego] Run prepared; CARLA was not started: {manifest_path}")
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), str(cfg["carla_agents_path"])] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env[CHILD_ENV] = str(manifest_path)
    cmd = [str(cfg["python311"]), "-m", "source.design_exploration.optimization.launch_parego"]
    print(f"[launch_parego] Run manifest: {manifest_path}")
    _run_optimizer_child(cmd, env=env)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[launch_parego] Interrupted; optimizer and simulator cleanup finished.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[launch_parego] Error: {exc}", file=sys.stderr)
        sys.exit(1)
