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
ParEGO Quantile Forest runner that stays in Python 3.11 and delegates simulation
work to a Python 3.8 process via NDJSON over stdio.

The 3.8 process runs `sim_ipc_server.py`, which owns the CARLA workers. This client:
  1. Spawns the 3.8 server.
  2. Runs ParEGO selection over the candidate set (the audited set) in 3.11.
  3. Sends simulation requests to 3.8 and consumes the returned payloads.

The on-wire protocol is described in sim_ipc_server.py.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pathlib import Path

from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.design_space import Design, SpaceSpec
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager, EpisodeMeta
from source.design_exploration.optimization.cost_model import (
    build_cost_model,
    cost_model_manifest,
)
from source.design_exploration.optimization.parego_qrf_common import (
    DEFAULT_RHO,
    ParEGOQuantileRF,
    _aggregate,
    _aggregate_with_noise,
    _extract_metric,
    _non_dominated_mask,
    _portfolio_noise,
    _payload_to_episode_result,
    resolve_parego_profile,
)

LOG = logging.getLogger("ipc.parego_qrf_client")


@dataclass(frozen=True)
class IPCScenarioSpec:
    name: str
    cfg_path: str
    episodes_root: Optional[str] = None


class NDJSONProcessClient:
    """Manages a child process that speaks NDJSON over stdin/stdout."""

    def __init__(self, cmd: List[str], ready_timeout: float = 30.0):
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            preexec_fn=os.setsid,
        )
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._closed = False
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        try:
            self._wait_for_ready(timeout=ready_timeout)
        except BaseException:
            self.shutdown(timeout=min(5.0, ready_timeout))
            raise

    def _wait_for_ready(self, timeout: float):
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                for rid, entry in list(self._pending.items()):
                    resp = entry.get("resp")
                    if resp and resp.get("ready"):
                        self._pending.pop(rid, None)
                        return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for server ready signal")

    def _read_loop(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            # Simulator occasionally prints non-JSON lines (e.g., stream deletion warnings).
            if not line.startswith("{") and not line.startswith("["):
                LOG.debug("Non-JSON line from child (ignored): %s", line)
                continue
            try:
                msg = json.loads(line)
            except Exception:
                LOG.error("Failed to parse line from child: %s", line)
                continue
            req_id = msg.get("id", "ready")
            with self._lock:
                entry = self._pending.get(req_id)
                if entry is None:
                    # Treat as unsolicited; store under a synthetic slot.
                    self._pending[req_id] = {"event": threading.Event(), "resp": msg}
                    continue
                entry["resp"] = msg
                if "event" in entry:
                    entry["event"].set()
        self._running = False

    def _drain_stderr(self):
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            text = line.rstrip()
            if "cannot parse georeference" in text:
                continue
            LOG.warning("[sim-server] %s", text)

    def request(self, payload: dict, timeout: float = 300.0) -> dict:
        if not self._running:
            raise RuntimeError("child process is not running")
        rid = uuid.uuid4().hex
        payload = dict(payload)
        payload["id"] = rid
        event = threading.Event()
        with self._lock:
            self._pending[rid] = {"event": event, "resp": None}
            if self.proc.stdin is None:
                raise RuntimeError("stdin closed")
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        if not event.wait(timeout):
            self._pending.pop(rid, None)
            raise TimeoutError(f"request {rid} timed out")
        resp = self._pending.pop(rid)["resp"]
        return resp

    def shutdown(self, timeout: float = 5.0):
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
            if self.proc.poll() is not None:
                self._running = False
                return

            graceful_request_completed = False
            try:
                self.request({"action": "shutdown"}, timeout=timeout)
                graceful_request_completed = True
            except BaseException:
                pass
            if graceful_request_completed:
                try:
                    self.proc.wait(timeout=max(15.0, timeout))
                    self._running = False
                    return
                except subprocess.TimeoutExpired:
                    pass

            try:
                pgid = os.getpgid(self.proc.pid)
            except (ProcessLookupError, OSError):
                pgid = None

            def _send(sig: int) -> None:
                try:
                    if pgid is not None:
                        os.killpg(pgid, sig)
                    else:
                        self.proc.send_signal(sig)
                except (ProcessLookupError, OSError):
                    pass

            for sig, wait_timeout in (
                (signal.SIGTERM, max(15.0, timeout)),
                (signal.SIGKILL, 5.0),
            ):
                if self.proc.poll() is not None:
                    break
                _send(sig)
                try:
                    self.proc.wait(timeout=wait_timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            self._running = False


class RemoteSimulator:
    """Thin wrapper that calls the 3.8 server."""

    def __init__(
        self,
        server_cmd: List[str],
        ready_timeout: float = 30.0,
        req_timeout: float = 300.0,
        *,
        worker_count: int = 1,
        seconds_per_episode: float = 120.0,
    ):
        self.client = NDJSONProcessClient(server_cmd, ready_timeout=ready_timeout)
        self.req_timeout = req_timeout
        self.worker_count = max(1, int(worker_count))
        self.seconds_per_episode = float(seconds_per_episode)
        self.last_accounting: Dict[str, Any] = {}

    def simulate(self, design: Design, scenario_seeds: Dict[str, List[int]]) -> Dict[str, List[EpisodeResult]]:
        req = {
            "action": "evaluate_design",
            "design": dict(design),
            "scenario_seeds": [
                {"name": name, "seeds": [int(seed) for seed in seeds]}
                for name, seeds in scenario_seeds.items()
            ],
        }
        total_episodes = sum(len(seeds) for seeds in scenario_seeds.values())
        estimated_batches = max(1, math.ceil(float(total_episodes) / float(self.worker_count)))
        req_timeout = max(self.req_timeout, self.seconds_per_episode * estimated_batches)
        resp = self.client.request(req, timeout=req_timeout)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown simulation error"))
        self.last_accounting = dict(resp.get("accounting") or {})
        episodes: Dict[str, List[EpisodeResult]] = {}
        episodes_resp = resp.get("episodes", {}) or {}
        for sc_name, requested_seeds in scenario_seeds.items():
            payloads = episodes_resp.get(sc_name, [])
            if not isinstance(payloads, list):
                raise IPCProtocolError(
                    f"Scenario '{sc_name}' returned malformed payload container of type {type(payloads).__name__}"
                )
            returned_seeds = [int(item.get("seed")) for item in payloads]
            if returned_seeds != [int(seed) for seed in requested_seeds]:
                raise IPCProtocolError(
                    f"Scenario '{sc_name}' returned seeds {returned_seeds}, "
                    f"expected {[int(seed) for seed in requested_seeds]}"
                )
            eps_list = []
            for item in payloads:
                if not isinstance(item, dict):
                    raise IPCProtocolError(
                        f"Scenario '{sc_name}' returned malformed payload entry of type {type(item).__name__}"
                    )
                if "payload" not in item:
                    raise IPCProtocolError(
                        f"Scenario '{sc_name}' returned payload entry without 'payload'"
                    )
                ep = EpisodeResult()
                ep.result = item.get("payload", {})
                eps_list.append(ep)
            episodes[sc_name] = eps_list
        return episodes

    def close(self):
        self.client.shutdown()


def _partitioned_episode_seed(
    offset: int,
    gidx: int,
    scenario_index: int,
    ordinal: int,
    *,
    scenario_count: int,
    repetitions: int,
) -> int:
    """Map every design/scenario/ordinal tuple injectively into a seed range."""
    if offset < 0:
        raise ValueError("episode seed offset must be non-negative")
    if gidx < 0 or scenario_index < 0 or ordinal < 0:
        raise ValueError("episode seed coordinates must be non-negative")
    if scenario_index >= scenario_count:
        raise ValueError("scenario index is outside the frozen scenario order")
    if ordinal >= repetitions:
        raise ValueError("episode ordinal exceeds the fixed replication budget")
    seed = int(offset) + (
        (int(gidx) * int(scenario_count) + int(scenario_index)) * int(repetitions)
        + int(ordinal)
    )
    if seed > 0xFFFFFFFF:
        raise ValueError("partitioned episode seed exceeds the uint32 range")
    return seed


class IPCProtocolError(RuntimeError):
    """Non-restartable client/server protocol or payload validation error."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp_path), str(path))


def _write_csv_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(list(rows)).to_csv(tmp_path, index=False)
    os.replace(str(tmp_path), str(path))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _same_optional_float(actual: Any, expected: Any) -> bool:
    if _is_missing(actual) and _is_missing(expected):
        return True
    if _is_missing(actual) or _is_missing(expected):
        return False
    return bool(np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12))


def _hypervolume_2d_min_strict(
    costs: np.ndarray, perfs: np.ndarray, ref: np.ndarray
) -> float:
    """Compute 2D minimization HV and reject a non-dominating reference."""
    costs = np.asarray(costs, dtype=float).reshape(-1)
    perfs = np.asarray(perfs, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    if costs.shape != perfs.shape:
        raise ValueError("hypervolume objective arrays must have equal length")
    if ref.shape != (2,) or not np.all(np.isfinite(ref)):
        raise ValueError("hypervolume reference must contain two finite values")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(perfs)):
        raise ValueError("hypervolume objectives must be finite")
    if costs.size == 0:
        return 0.0
    outside = (costs > ref[0]) | (perfs > ref[1])
    if np.any(outside):
        raise ValueError(
            "hypervolume reference does not weakly dominate every point: "
            "max_cost=%.12g ref_cost=%.12g max_perf=%.12g ref_perf=%.12g"
            % (costs.max(), ref[0], perfs.max(), ref[1])
        )
    order = np.argsort(costs)
    hv = 0.0
    prev_y = float(ref[1])
    for idx in order:
        x = float(costs[idx])
        y = float(perfs[idx])
        hv += (float(ref[0]) - x) * max(0.0, prev_y - y)
        prev_y = min(prev_y, y)
    return float(hv)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no", ""):
            return False
    return bool(value)


def _compute_operational_gate_status(
    results_log: Sequence[Dict[str, Any]],
    *,
    prefix_rows: int,
    window_designs: int,
    episodes_per_scenario: int,
    scenario_count: int,
    max_seconds_per_paired_trial: Optional[float],
    max_retries_per_100_episodes: Optional[float],
) -> Dict[str, Any]:
    """Evaluate the rolling gate without mutating or launching campaign state."""
    if prefix_rows < 0 or prefix_rows > len(results_log):
        raise ValueError("operational gate prefix is outside the campaign history")
    new_rows = list(results_log[prefix_rows:])
    status: Dict[str, Any] = {
        "schema_version": 1,
        "status": "warming" if len(new_rows) < window_designs else "passed",
        "prefix_rows_excluded": int(prefix_rows),
        "new_designs": int(len(new_rows)),
        "window_designs": int(window_designs),
        "max_seconds_per_paired_trial": max_seconds_per_paired_trial,
        "max_retries_per_100_episodes": max_retries_per_100_episodes,
        "stop_required": False,
    }
    if len(new_rows) < window_designs:
        return status

    selected = new_rows[-window_designs:]
    paired_trials = int(window_designs) * int(episodes_per_scenario)
    required_episodes = paired_trials * int(scenario_count)
    seconds_per_paired = (
        sum(float(row.get("time", 0.0)) for row in selected) / paired_trials
    )
    retries = sum(
        int(float(row.get("ipc_retried_chunks", 0) or 0)) for row in selected
    )
    retries_per_100 = 100.0 * retries / required_episodes
    breaches = []
    if (
        max_seconds_per_paired_trial is not None
        and seconds_per_paired > float(max_seconds_per_paired_trial)
    ):
        breaches.append("throughput")
    if (
        max_retries_per_100_episodes is not None
        and retries_per_100 > float(max_retries_per_100_episodes)
    ):
        breaches.append("retry_rate")
    status.update(
        {
            "status": "stopped" if breaches else "passed",
            "seconds_per_paired_trial": seconds_per_paired,
            "retried_chunks": retries,
            "required_episodes": required_episodes,
            "retries_per_100_episodes": retries_per_100,
            "breaches": breaches,
            "stop_required": bool(breaches),
            "window_gidx": [int(row["gidx"]) for row in selected],
        }
    )
    return status


def replay_optimizer_history(
    optimizer: ParEGOQuantileRF,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Replay a persisted prefix and prove that every historical proposal recurs.

    Calling ``ask`` before each stored ``tell`` reconstructs the RNG state.  The
    global design index, named policy, weights and acquisition diagnostics are
    checked before the stored objectives are restored.
    """
    accepted = 0
    eval_costs: List[float] = []
    eval_perfs: List[float] = []
    for position, row in enumerate(rows):
        expected_iter = int(row.get("iter", position))
        if expected_iter != position:
            raise RuntimeError(
                f"Continuation history is not contiguous at row {position}: iter={expected_iter}."
            )
        proposed = optimizer.ask()
        expected_gidx = int(row["gidx"])
        if proposed != expected_gidx:
            raise RuntimeError(
                "Deterministic continuation check failed at iteration "
                f"{position}: optimizer proposed gidx {proposed}, history requires {expected_gidx}."
            )
        for field, actual in (
            ("profile", optimizer.profile),
            ("acquisition", optimizer.acquisition),
        ):
            expected = row.get(field)
            if not _is_missing(expected) and str(expected) != str(actual):
                raise RuntimeError(
                    f"Continuation history {field} mismatch at iteration {position}: "
                    f"{expected!r} != {actual!r}."
                )
        weights_raw = row.get("scalarization_weights")
        if not _is_missing(weights_raw):
            expected_weights = np.asarray(
                json.loads(weights_raw) if isinstance(weights_raw, str) else weights_raw,
                dtype=float,
            )
            if optimizer.last_weights is None or not np.allclose(
                optimizer.last_weights, expected_weights, rtol=0.0, atol=1e-15
            ):
                raise RuntimeError(
                    f"Continuation scalarization weights mismatch at iteration {position}."
                )
        for field, actual in (
            ("acq_value", optimizer.last_acq),
        ):
            expected = row.get(field)
            if not _same_optional_float(actual, expected):
                raise RuntimeError(
                    f"Continuation {field} mismatch at iteration {position}: "
                    f"{expected!r} != {actual!r}."
                )
        for field, actual in (
            ("surrogate_quantile_span", optimizer.last_surrogate_quantile_span),
            ("acquisition_score_span", optimizer.last_acquisition_score_span),
            ("tie_candidate_count", optimizer.last_tie_candidate_count),
            (
                "forest_split_tree_fraction",
                optimizer.last_forest_split_tree_fraction,
            ),
            ("forest_mean_depth", optimizer.last_forest_mean_depth),
            ("forest_max_depth", optimizer.last_forest_max_depth),
        ):
            expected = row.get(field)
            if not _is_missing(expected) and not _same_optional_float(actual, expected):
                raise RuntimeError(
                    f"Continuation {field} mismatch at iteration {position}: "
                    f"{expected!r} != {actual!r}."
                )
        for field, actual in (
            (
                "surrogate_prediction_degenerate",
                optimizer.last_surrogate_prediction_degenerate,
            ),
            (
                "acquisition_score_degenerate",
                optimizer.last_acquisition_score_degenerate,
            ),
        ):
            expected = row.get(field)
            if not _is_missing(expected) and _as_bool(expected) != bool(actual):
                raise RuntimeError(
                    f"Continuation {field} mismatch at iteration {position}."
                )
        for field, actual in (
            ("tie_break_policy", optimizer.last_tie_break_policy),
            ("selection_reason", optimizer.last_selection_reason),
        ):
            expected = row.get(field)
            if not _is_missing(expected) and str(expected) != str(actual):
                raise RuntimeError(
                    f"Continuation {field} mismatch at iteration {position}."
                )

        obs_var_raw = row.get("obs_var")
        obs_var = None if _is_missing(obs_var_raw) else float(obs_var_raw)
        obj_cost = float(row["obj_cost"])
        obj_perf = float(row["obj_perf"])
        optimizer.tell(expected_gidx, [obj_cost, obj_perf], noise_variance=obs_var)
        accepted += 1
        eval_costs.append(obj_cost)
        eval_perfs.append(obj_perf)
    return {
        "attempts": len(rows),
        "accepted": accepted,
        "eval_costs": eval_costs,
        "eval_perfs": eval_perfs,
    }


def _is_non_restartable_sim_error(exc: Exception) -> bool:
    if isinstance(exc, IPCProtocolError):
        return True
    msg = str(exc).lower()
    markers = (
        "duplicate seed",
        "missing payload",
        "malformed payload",
        "payload entry without",
        "returned seeds",
        "seed list length mismatch",
        "worker returned",
        "protocol",
        "validation",
    )
    return any(marker in msg for marker in markers)


def run_parego_over_candidates(
    space_spec: SpaceSpec,
    scenario_specs: Sequence[IPCScenarioSpec],
    python38: str,
    candidate_indices: np.ndarray,
    out_dir: str,
    *,
    cost_profile_config: str,
    cost_profile_id: str,
    hv_ref_point: Sequence[float],
    episode_seed_offset: int,
    carla_path: Optional[str] = None,
    budget: int = 30,
    audited_count: Optional[int] = None,
    warm_start_episode_roots: Optional[Sequence[str]] = None,
    perf_metric: str = "kpi.vel_at_min_d",
    episodes_per_scenario: int = 20,
    scenario_agg: str = "mean",
    portfolio_agg: str = "mean",
    maximize_metric: bool = True,
    random_seed: Optional[int] = None,
    rf_quantile: float = 0.25,
    rf_estimators: int = 200,
    rf_min_samples_leaf: int = 3,
    rho: float = DEFAULT_RHO,
    profile: Optional[str] = None,
    num_carla_instances: Optional[int] = None,
    hv_reference_metadata: Optional[Mapping[str, Any]] = None,
    episode_chunk_size: int = 1,
    max_in_flight: Optional[int] = None,
    operational_gate_window_designs: int = 5,
    max_seconds_per_paired_trial: Optional[float] = None,
    max_retries_per_100_episodes: Optional[float] = None,
    resume: bool = False,
):
    profile_spec = resolve_parego_profile(profile)
    os.makedirs(out_dir, exist_ok=True)
    return _run_parego_qrf_via_ipc_common(
        space_spec=space_spec,
        scenario_specs=scenario_specs,
        python38=python38,
        carla_path=carla_path,
        candidate_indices=np.asarray(candidate_indices, dtype=int),
        budget=budget,
        audited_count=audited_count,
        out_dir=out_dir,
        warm_start_episode_roots=warm_start_episode_roots,
        perf_metric=perf_metric,
        episodes_per_scenario=episodes_per_scenario,
        scenario_agg=scenario_agg,
        portfolio_agg=portfolio_agg,
        maximize_metric=maximize_metric,
        random_seed=random_seed,
        rf_quantile=rf_quantile,
        rf_estimators=rf_estimators,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rho=rho,
        profile=profile_spec.profile,
        num_carla_instances=num_carla_instances,
        cost_profile_config=cost_profile_config,
        cost_profile_id=cost_profile_id,
        hv_ref_point=hv_ref_point,
        hv_reference_metadata=hv_reference_metadata,
        episode_chunk_size=episode_chunk_size,
        max_in_flight=max_in_flight,
        episode_seed_offset=episode_seed_offset,
        operational_gate_window_designs=operational_gate_window_designs,
        max_seconds_per_paired_trial=max_seconds_per_paired_trial,
        max_retries_per_100_episodes=max_retries_per_100_episodes,
        resume=resume,
    )


def _run_parego_qrf_via_ipc_common(
    *,
    space_spec: SpaceSpec,
    scenario_specs: Sequence[IPCScenarioSpec],
    python38: str,
    carla_path: Optional[str],
    candidate_indices: np.ndarray,
    budget: int,
    audited_count: Optional[int],
    out_dir: str,
    warm_start_episode_roots: Optional[Sequence[str]],
    perf_metric: str,
    episodes_per_scenario: int,
    scenario_agg: str,
    portfolio_agg: str,
    maximize_metric: bool,
    random_seed: Optional[int],
    rf_quantile: float,
    rf_estimators: int,
    rf_min_samples_leaf: int,
    rho: float,
    profile: str,
    num_carla_instances: Optional[int],
    cost_profile_config: str,
    cost_profile_id: str,
    hv_ref_point: Sequence[float],
    hv_reference_metadata: Optional[Mapping[str, Any]],
    episode_chunk_size: int,
    max_in_flight: Optional[int],
    episode_seed_offset: int,
    operational_gate_window_designs: int,
    max_seconds_per_paired_trial: Optional[float],
    max_retries_per_100_episodes: Optional[float],
    resume: bool,
):
    if not scenario_specs:
        raise ValueError("No scenario specs provided")
    profile_spec = resolve_parego_profile(profile)
    n_initial_points = profile_spec.n_initial_points
    algo_version = f"parego_qrf_ipc_{profile_spec.profile}"
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not np.isfinite(float(rho)) or float(rho) < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if episodes_per_scenario <= 0:
        raise ValueError("episodes_per_scenario must be positive")
    if episode_chunk_size <= 0:
        raise ValueError("episode_chunk_size must be positive")
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive when provided")
    if episode_seed_offset is None:
        raise ValueError("episode_seed_offset is required")
    if operational_gate_window_designs <= 0:
        raise ValueError("operational_gate_window_designs must be positive")
    if (
        max_seconds_per_paired_trial is not None
        and max_seconds_per_paired_trial <= 0
    ):
        raise ValueError("max_seconds_per_paired_trial must be positive")
    if (
        max_retries_per_100_episodes is not None
        and max_retries_per_100_episodes < 0
    ):
        raise ValueError("max_retries_per_100_episodes must be non-negative")
    if budget < n_initial_points:
        raise ValueError("budget must be at least n_initial_points")
    if hv_ref_point is None or len(hv_ref_point) != 2:
        raise ValueError("hv_ref_point must be a sequence of two floats [cost_ref, perf_ref]")

    warm_start_episode_roots = [str(path) for path in (warm_start_episode_roots or []) if path]
    resolved_carla_path = str(carla_path or os.environ.get("CARLA_PATH") or "")
    if not resolved_carla_path:
        raise ValueError("A CARLA executable is required for IPC simulation.")

    GRID = space_spec.build_grid()
    g_min = GRID.min(axis=0)
    g_max = GRID.max(axis=0)
    g_range = np.where(g_max > g_min, g_max - g_min, 1.0)
    GRID_NORM = (GRID - g_min) / g_range
    cost_model = build_cost_model(
        space_spec,
        config_path=cost_profile_config,
        profile_id=cost_profile_id,
    )
    frozen_cost_model = cost_model_manifest(cost_model, space_spec)
    run_config = {
        "algo_version": algo_version,
        "profile": profile_spec.profile,
        "resolved_policy": {
            "acquisition": profile_spec.acquisition,
            "n_initial_points": profile_spec.n_initial_points,
            "surrogate_target": profile_spec.surrogate_target,
            "cost_normalization": profile_spec.cost_normalization,
            "performance_normalization": profile_spec.performance_normalization,
            "tie_break": profile_spec.tie_break,
            "degenerate_prediction_policy": (
                profile_spec.degenerate_prediction_policy
            ),
        },
        "budget": int(budget),
        "episodes_per_scenario": int(episodes_per_scenario),
        "perf_metric": str(perf_metric),
        "scenario_agg": str(scenario_agg),
        "portfolio_agg": str(portfolio_agg),
        "maximize_metric": bool(maximize_metric),
        "random_seed": random_seed,
        "rf_quantile": float(rf_quantile),
        "rf_estimators": int(rf_estimators),
        "rf_min_samples_leaf": int(rf_min_samples_leaf),
        "rho": float(rho),
        "cost_model": frozen_cost_model,
        "hv_ref_point": [float(value) for value in hv_ref_point],
        "hypervolume_reference": (
            dict(hv_reference_metadata)
            if hv_reference_metadata is not None
            else None
        ),
        "scenarios": [spec.name for spec in scenario_specs],
        "carla_path": resolved_carla_path,
        "episode_chunk_size": int(episode_chunk_size),
        "max_in_flight": (
            int(max_in_flight) if max_in_flight is not None else None
        ),
        "episode_seed_policy": "partitioned_v1",
        "episode_seed_offset": int(episode_seed_offset),
        "operational_gate": {
            "window_designs": int(operational_gate_window_designs),
            "max_seconds_per_paired_trial": max_seconds_per_paired_trial,
            "max_retries_per_100_episodes": max_retries_per_100_episodes,
        },
        "resume": bool(resume),
        "scenario_inputs": [
            {"name": spec.name, "cfg_path": str(Path(spec.cfg_path).resolve())}
            for spec in scenario_specs
        ],
    }
    os.makedirs(out_dir, exist_ok=True)

    def _row_seed(row: Dict[str, Any]) -> Optional[int]:
        payload = row.get("payload", {}) or {}
        if not isinstance(payload, dict):
            return None
        meta = payload.get("meta", {}) or {}
        seed_val = meta.get("seed")
        if seed_val is None:
            return None
        try:
            return int(seed_val)
        except Exception:
            return None

    def _row_signature(row: Dict[str, Any]) -> str:
        seed_val = _row_seed(row)
        if seed_val is not None:
            return f"seed:{seed_val}"
        payload = row.get("payload", {}) or {}
        try:
            return "payload:" + json.dumps(payload, sort_keys=True, default=str, allow_nan=True)
        except Exception:
            return "payload:" + str(payload)

    def _build_cache_sources(spec: IPCScenarioSpec) -> List[EpisodeManager]:
        sources: List[EpisodeManager] = []
        seen: set[str] = set()
        for root in [*warm_start_episode_roots, spec.episodes_root]:
            if not root:
                continue
            root_abs = os.path.abspath(str(root))
            if root_abs in seen:
                continue
            seen.add(root_abs)
            sources.append(EpisodeManager(root_abs, spec.name, space_spec, GRID))
        return sources

    def _cached_rows_for_gidx(spec_name: str, gidx: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for manager in cache_sources[spec_name]:
            for row in manager.episodes(gidx, consume=False):
                sig = _row_signature(row)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                rows.append(row)
        return rows

    server_script = os.path.join(
        os.path.dirname(__file__),
        "sim_ipc_server.py",
    )
    server_cmd = [python38, server_script, "--algo-version", algo_version]
    server_cmd += ["--carla-path", resolved_carla_path]
    sim_out_dir = os.path.join(out_dir, "sim_out_ipc")
    os.makedirs(sim_out_dir, exist_ok=True)
    server_cmd += ["--sim-out", sim_out_dir]
    server_cmd += ["--episode-chunk-size", str(int(episode_chunk_size))]
    if max_in_flight is not None:
        server_cmd += ["--max-in-flight", str(int(max_in_flight))]
    if num_carla_instances is not None:
        server_cmd += ["--num-carla", str(num_carla_instances)]
    for spec in scenario_specs:
        server_cmd += ["--scenario", f"{spec.name}:{spec.cfg_path}"]

    sim: Optional[RemoteSimulator] = None

    def _simulate_with_retry(design_obj: Design, scenario_seed_map: Dict[str, List[int]], max_restarts: int = 3):
        nonlocal sim
        last_err = None
        for attempt in range(max_restarts + 1):
            try:
                if sim is None:
                    raise RuntimeError("simulation server has not been started")
                return sim.simulate(design_obj, scenario_seed_map)
            except Exception as exc:
                last_err = exc
                if _is_non_restartable_sim_error(exc):
                    LOG.error(
                        "Non-restartable simulation protocol/data error for gidx request; "
                        "scenarios=%s | episodes=%s | error=%s",
                        list(scenario_seed_map.keys()),
                        {name: len(seeds) for name, seeds in scenario_seed_map.items()},
                        exc,
                    )
                    raise
                try:
                    if sim is not None:
                        sim.close()
                except Exception:
                    pass
                if attempt >= max_restarts:
                    raise last_err
                # Back off before spawning a new server to give GPU/OS time to release
                # resources from crashed CARLA instances (exit code 139 = SIGSEGV).
                backoff_s = 30.0 * (attempt + 1)  # 30s, 60s, 90s
                LOG.warning(
                    "Simulate failed (attempt %d/%d); waiting %.0fs before restarting sim server... error=%s",
                    attempt + 1,
                    max_restarts + 1,
                    backoff_s,
                    exc,
                )
                time.sleep(backoff_s)
                sim = RemoteSimulator(server_cmd, worker_count=max(1, int(num_carla_instances or 1)))

    LOG.info("Full Design Space Size: %d", len(GRID))
    candidate_mask = np.zeros(len(GRID), dtype=bool)

    candidate_indices = np.unique(np.asarray(candidate_indices, dtype=int).reshape(-1))
    if candidate_indices.size and (candidate_indices.min() < 0 or candidate_indices.max() >= len(GRID)):
        raise ValueError("candidate_indices contain designs outside the design grid")
    candidate_mask[candidate_indices] = True
    LOG.info(
        "Candidate set: %d designs (%.2f%%)%s",
        len(candidate_indices),
        100 * len(candidate_indices) / len(GRID),
        "" if audited_count is None else f" of {int(audited_count)} audited",
    )

    X_candidates = GRID_NORM[candidate_indices]
    if len(candidate_indices) == 0:
        raise RuntimeError("Candidate set is empty; cannot run optimization.")

    np.save(os.path.join(out_dir, "candidate_mask.npy"), candidate_mask.astype(bool))
    np.save(os.path.join(out_dir, "candidate_indices.npy"), candidate_indices.astype(int))

    scenario_index = {
        spec.name: position for position, spec in enumerate(scenario_specs)
    }
    max_seed = _partitioned_episode_seed(
        int(episode_seed_offset),
        len(GRID) - 1,
        len(scenario_specs) - 1,
        episodes_per_scenario - 1,
        scenario_count=len(scenario_specs),
        repetitions=episodes_per_scenario,
    )
    LOG.info(
        "Using collision-free partitioned episode seeds [%d, %d].",
        int(episode_seed_offset),
        max_seed,
    )

    def _plan_seed_batch(gidx: int, scenario_name: str, start_ordinal: int, count: int) -> List[int]:
        return [
            _partitioned_episode_seed(
                int(episode_seed_offset),
                int(gidx),
                scenario_index[scenario_name],
                start_ordinal + offset,
                scenario_count=len(scenario_specs),
                repetitions=episodes_per_scenario,
            )
            for offset in range(int(count))
        ]

    # Episode caches per scenario (warm start / reuse across iterations)
    ep_managers = {}
    cache_sources: Dict[str, List[EpisodeManager]] = {}
    for spec in scenario_specs:
        root = spec.episodes_root
        if not root:
            raise ValueError(f"episodes_root missing for scenario '{spec.name}' in IPCScenarioSpec.")
        os.makedirs(root, exist_ok=True)
        ep_managers[spec.name] = EpisodeManager(root, spec.name, space_spec, GRID)
        cache_sources[spec.name] = _build_cache_sources(spec)
        cache_counts: Dict[int, int] = {}
        for manager in cache_sources[spec.name]:
            for gi, cnt in manager.list_gidx_with_counts().items():
                cache_counts[int(gi)] = cache_counts.get(int(gi), 0) + int(cnt)
        LOG.info(
            "Cache warm-start for scenario %s: %d episodes across %d designs from %d roots",
            spec.name,
            sum(cache_counts.values()),
            len(cache_counts),
            len(cache_sources[spec.name]),
        )

    def _verify_saved_objectives(rows: Sequence[Dict[str, Any]]) -> None:
        for position, row in enumerate(rows):
            gidx = int(row["gidx"])
            design_vec = GRID[gidx]
            design = Design({k: value for k, value in zip(space_spec.keys, design_vec)})
            expected_cost = float(cost_model.evaluate(design))
            if not _same_optional_float(row.get("obj_cost"), expected_cost):
                raise RuntimeError(
                    f"Continuation cost objective mismatch at iteration {position}."
                )
            for key, value in zip(space_spec.keys, design_vec):
                if key in row and not _is_missing(row[key]):
                    if not _same_optional_float(row[key], value):
                        raise RuntimeError(
                            f"Continuation design field '{key}' mismatch at iteration {position}."
                        )

            scenario_metrics: List[float] = []
            scenario_variances: List[Optional[float]] = []
            for spec in scenario_specs:
                cached_rows = _cached_rows_for_gidx(spec.name, gidx)
                values = [
                    _extract_metric(
                        _payload_to_episode_result(cached["payload"]), perf_metric
                    )
                    for cached in cached_rows
                ]
                finite = [
                    float(value)
                    for value in values
                    if value is not None and np.isfinite(value)
                ][:episodes_per_scenario]
                if len(finite) != episodes_per_scenario:
                    raise RuntimeError(
                        f"Continuation objective proof found {len(finite)}/"
                        f"{episodes_per_scenario} finite episodes for iteration {position}, "
                        f"gidx {gidx}, scenario {spec.name}."
                    )
                metric, variance = _aggregate_with_noise(finite, scenario_agg)
                scenario_metrics.append(metric)
                scenario_variances.append(variance)

            portfolio_metric = _aggregate(scenario_metrics, portfolio_agg)
            obj_perf = -portfolio_metric if maximize_metric else portfolio_metric
            obs_var = _portfolio_noise(scenario_variances, portfolio_agg)
            for label, actual, expected in (
                ("metric_agg", row.get("metric_agg"), portfolio_metric),
                ("obj_perf", row.get("obj_perf"), obj_perf),
                ("obs_var", row.get("obs_var"), obs_var),
            ):
                if not _same_optional_float(actual, expected):
                    raise RuntimeError(
                        f"Continuation {label} mismatch at iteration {position}: "
                        f"{actual!r} != {expected!r}."
                    )

    optimizer = ParEGOQuantileRF(
        X_candidates,
        candidate_indices,
        num_objectives=2,
        random_seed=random_seed,
        n_estimators=rf_estimators,
        min_samples_leaf=rf_min_samples_leaf,
        quantile=rf_quantile,
        rho=float(rho),
        profile=profile_spec.profile,
    )

    out_path = Path(out_dir)
    config_path = out_path / "parego_qrf_ipc_config.json"
    checkpoint_path = out_path / "parego_qrf_ipc_checkpoint.json"
    history_path = out_path / "parego_qrf_ipc_history.csv"
    hv_path = out_path / "hypervolume_log.csv"
    frontier_path = out_path / "parego_qrf_ipc_frontier.csv"

    previous_config: Dict[str, Any] = {}
    if resume and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        immutable_keys = (
            "algo_version",
            "profile",
            "resolved_policy",
            "episodes_per_scenario",
            "perf_metric",
            "scenario_agg",
            "portfolio_agg",
            "maximize_metric",
            "random_seed",
            "rf_quantile",
            "rf_estimators",
            "rf_min_samples_leaf",
            "rho",
            "cost_model",
            "hv_ref_point",
            "hypervolume_reference",
            "scenarios",
            "carla_path",
            "scenario_inputs",
            "episode_seed_policy",
            "episode_seed_offset",
            "operational_gate",
        )
        mismatches = [
            key
            for key in immutable_keys
            if key in previous_config and previous_config.get(key) != run_config.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Resume configuration changes immutable campaign fields: "
                + ", ".join(mismatches)
            )

    results_log: List[Dict[str, Any]] = []
    hv_log: List[Dict[str, Any]] = []
    checkpoint_source = "new"
    checkpoint_gate_prefix: Optional[int] = None
    if resume:
        if checkpoint_path.is_file():
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if int(checkpoint.get("schema_version", -1)) != 1:
                raise RuntimeError("Unsupported ParEGO checkpoint schema.")
            results_log = list(checkpoint.get("history") or [])
            hv_log = list(checkpoint.get("hypervolume") or [])
            if checkpoint.get("operational_gate_baseline_rows") is not None:
                checkpoint_gate_prefix = int(
                    checkpoint["operational_gate_baseline_rows"]
                )
            checkpoint_source = "checkpoint"
        else:
            raise RuntimeError(
                f"--resume was requested but no checkpoint exists in {out_dir}."
            )
    elif checkpoint_path.exists() or history_path.exists():
        raise RuntimeError(
            f"ParEGO output directory already contains campaign state: {out_dir}. "
            "Pass resume=True to continue it."
        )

    if resume:
        _verify_saved_objectives(results_log)
    replay_state = replay_optimizer_history(optimizer, results_log)
    accepted = int(replay_state["accepted"])
    attempts = int(replay_state["attempts"])
    eval_costs: List[float] = list(replay_state["eval_costs"])
    eval_perfs: List[float] = list(replay_state["eval_perfs"])
    if accepted > budget:
        raise RuntimeError(
            f"Checkpoint already contains {accepted} accepted designs, exceeding budget={budget}."
        )
    run_config["continuation"] = {
        "source": checkpoint_source,
        "prefix_attempts": attempts,
        "prefix_accepted": accepted,
    }
    config_gate_prefix = previous_config.get("operational_gate_baseline_rows")
    if (
        checkpoint_gate_prefix is not None
        and config_gate_prefix is not None
        and checkpoint_gate_prefix != int(config_gate_prefix)
    ):
        raise RuntimeError(
            "Operational-gate baseline differs between config and checkpoint."
        )
    previous_gate_prefix = (
        checkpoint_gate_prefix
        if checkpoint_gate_prefix is not None
        else config_gate_prefix
    )
    gate_prefix_rows = (
        int(previous_gate_prefix)
        if previous_gate_prefix is not None
        else (len(results_log) if resume else 0)
    )
    if gate_prefix_rows < 0 or gate_prefix_rows > len(results_log):
        raise RuntimeError(
            "Persisted operational-gate baseline is outside campaign history."
        )
    run_config["operational_gate_baseline_rows"] = gate_prefix_rows
    _write_json_atomic(config_path, run_config)
    gate_path = out_path / "parego_operational_gate.json"

    def _operational_gate_status() -> Dict[str, Any]:
        return _compute_operational_gate_status(
            results_log,
            prefix_rows=gate_prefix_rows,
            window_designs=int(operational_gate_window_designs),
            episodes_per_scenario=int(episodes_per_scenario),
            scenario_count=len(scenario_specs),
            max_seconds_per_paired_trial=max_seconds_per_paired_trial,
            max_retries_per_100_episodes=max_retries_per_100_episodes,
        )

    _write_json_atomic(gate_path, _operational_gate_status())

    def _frontier_rows() -> List[Dict[str, Any]]:
        valid = list(results_log)
        if not valid:
            return []
        costs = np.asarray([float(row["obj_cost"]) for row in valid], dtype=float)
        perfs = np.asarray([float(row["obj_perf"]) for row in valid], dtype=float)
        mask = _non_dominated_mask(costs, perfs)
        return [row for row, keep in zip(valid, mask) if bool(keep)]

    if resume and frontier_path.is_file():
        recorded_frontier = pd.read_csv(frontier_path)
        replayed_frontier = pd.DataFrame(_frontier_rows())
        recorded_ids = recorded_frontier.get("gidx", pd.Series(dtype=int)).astype(int).tolist()
        replayed_ids = replayed_frontier.get("gidx", pd.Series(dtype=int)).astype(int).tolist()
        if recorded_ids != replayed_ids:
            raise RuntimeError(
                "Deterministic continuation frontier mismatch: "
                f"recorded {recorded_ids}, replayed {replayed_ids}."
            )
        for objective in ("obj_cost", "obj_perf"):
            if objective in recorded_frontier and objective in replayed_frontier:
                if not np.allclose(
                    recorded_frontier[objective].to_numpy(dtype=float),
                    replayed_frontier[objective].to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Deterministic continuation frontier {objective} mismatch."
                    )

    def _persist_campaign_state() -> None:
        _write_csv_atomic(history_path, results_log)
        _write_csv_atomic(hv_path, hv_log)
        _write_csv_atomic(frontier_path, _frontier_rows())
        _write_json_atomic(
            checkpoint_path,
            {
                "schema_version": 1,
                "budget": int(budget),
                "attempts": int(attempts),
                "accepted": int(accepted),
                "operational_gate_baseline_rows": int(gate_prefix_rows),
                "history": results_log,
                "hypervolume": hv_log,
            },
        )

    _persist_campaign_state()
    LOG.info(
        "ParEGO continuation proof passed: %d attempts / %d accepted from %s.",
        attempts,
        accepted,
        checkpoint_source,
    )
    ref_point = np.asarray(hv_ref_point, dtype=float)
    if not np.all(np.isfinite(ref_point)):
        raise ValueError("hv_ref_point must contain only finite values")
    LOG.info("Hypervolume reference point: cost_ref=%.4f, perf_ref=%.4f", *ref_point)

    if accepted < budget:
        LOG.info("Starting 3.8 sim server: %s", " ".join(server_cmd))
        sim = RemoteSimulator(
            server_cmd, worker_count=max(1, int(num_carla_instances or 1))
        )
        LOG.info(
            "Starting ParEGO QRF (IPC) toward budget %d from accepted prefix %d...",
            budget,
            accepted,
        )
    else:
        LOG.info(
            "ParEGO campaign already satisfies budget=%d; simulator will not start.",
            budget,
        )

    try:
        max_attempts = budget * 5  # safety bound to avoid infinite loops
        while accepted < budget and attempts < max_attempts:
            attempts += 1
            iter_start = time.time()
            gidx = optimizer.ask()
            if gidx is None:
                LOG.info("Stopping early: no new safe designs to evaluate.")
                break
            acquisition_fields = {
                "profile": optimizer.profile,
                "acquisition": optimizer.acquisition,
                "acq_value": optimizer.last_acq,
                "scalarization_weights": (
                    json.dumps(optimizer.last_weights.tolist())
                    if optimizer.last_weights is not None
                    else None
                ),
                "surrogate_quantile_span": optimizer.last_surrogate_quantile_span,
                "surrogate_prediction_degenerate": (
                    optimizer.last_surrogate_prediction_degenerate
                ),
                "acquisition_score_span": optimizer.last_acquisition_score_span,
                "acquisition_score_degenerate": (
                    optimizer.last_acquisition_score_degenerate
                ),
                "tie_candidate_count": optimizer.last_tie_candidate_count,
                "tie_break_policy": optimizer.last_tie_break_policy,
                "selection_reason": optimizer.last_selection_reason,
                "forest_split_tree_fraction": (
                    optimizer.last_forest_split_tree_fraction
                ),
                "forest_mean_depth": optimizer.last_forest_mean_depth,
                "forest_max_depth": optimizer.last_forest_max_depth,
            }

            design_vec = GRID[gidx]
            design_dict = {k: val for k, val in zip(space_spec.keys, design_vec)}
            design = Design(design_dict)

            obj_cost = cost_model.evaluate(design)

            perf_vals_map: Dict[str, List[float]] = {}
            cache_count_map: Dict[str, int] = {}
            missing_reqs = []
            simulation_accounting: Dict[str, Any] = {}

            for spec in scenario_specs:
                ep_man = ep_managers[spec.name]
                cached_rows = _cached_rows_for_gidx(spec.name, gidx)
                cache_count_map[spec.name] = len(cached_rows)
                perf_vals = [
                    _extract_metric(_payload_to_episode_result(row["payload"]), perf_metric)
                    for row in cached_rows
                ]
                perf_vals = [v for v in perf_vals if v is not None and np.isfinite(v)]
                # Fixed replication means exactly R finite observations enter
                # every scenario aggregate.  Previously a warm cache silently
                # changed the effective budget from design to design.
                perf_vals_map[spec.name] = perf_vals[:episodes_per_scenario]

                missing = max(0, episodes_per_scenario - len(perf_vals_map[spec.name]))
                if missing > 0:
                    missing_reqs.append((spec, missing))

            if missing_reqs:
                seed_plan_map = {
                    spec.name: _plan_seed_batch(gidx, spec.name, cache_count_map[spec.name], missing)
                    for spec, missing in missing_reqs
                }
                episodes_map = _simulate_with_retry(design, seed_plan_map)
                if sim is not None:
                    simulation_accounting = dict(sim.last_accounting)
                for spec, _ in missing_reqs:
                    ep_man = ep_managers[spec.name]
                    new_eps = episodes_map.get(spec.name, [])
                    if new_eps:
                        meta = EpisodeMeta(scenario=spec.name, split="optimization", provenance="ParegoQRF")
                        ep_man.append_batch(gidx, new_eps, meta)
                        ep_man.flush()
                        cache_count_map[spec.name] += len(new_eps)
                        perf_vals_map[spec.name].extend(
                            [
                                v
                                for v in (
                                    _extract_metric(_payload_to_episode_result(ep.result), perf_metric)
                                    for ep in new_eps
                                )
                                if v is not None and np.isfinite(v)
                            ]
                        )
                        perf_vals_map[spec.name] = perf_vals_map[spec.name][
                            :episodes_per_scenario
                        ]

            sc_perfs = []
            sc_vars: Dict[str, Optional[float]] = {}
            sc_counts: Dict[str, int] = {}
            missing_data = False
            for spec in scenario_specs:
                perf_vals = perf_vals_map.get(spec.name, [])
                sc_counts[spec.name] = len(perf_vals)
                if len(perf_vals) != episodes_per_scenario:
                    LOG.error(
                        "Design %d has %d/%d finite %s observations for %s; "
                        "refusing a variable-budget objective.",
                        gidx,
                        len(perf_vals),
                        episodes_per_scenario,
                        perf_metric,
                        spec.name,
                    )
                    missing_data = True
                    break
                sc_metric, sc_var = _aggregate_with_noise(perf_vals, scenario_agg)
                sc_vars[spec.name] = sc_var
                if not np.isfinite(sc_metric):
                    missing_data = True
                    break
                sc_perfs.append(sc_metric)

            if missing_data or not sc_perfs:
                raise RuntimeError(
                    f"Design {gidx} did not produce the fixed finite episode budget; "
                    "the completed campaign prefix remains checkpointed for resume."
                )

            if portfolio_agg == "mean":
                portfolio_metric = _aggregate(sc_perfs, "mean")
            else:
                portfolio_metric = _aggregate(sc_perfs, portfolio_agg)

            obj_perf = -portfolio_metric if maximize_metric else portfolio_metric
            obs_var = _portfolio_noise(
                [sc_vars.get(spec.name) for spec in scenario_specs], portfolio_agg
            )
            optimizer.tell(gidx, [obj_cost, obj_perf], noise_variance=obs_var)

            row = {
                "iter": attempts - 1,
                "gidx": int(gidx),
                "obj_cost": obj_cost,
                "metric_agg": portfolio_metric,
                "obj_perf": obj_perf,
                "time": time.time() - iter_start,
                "obs_var": obs_var,
                **acquisition_fields,
                **{f"n_eps_{name}": count for name, count in sc_counts.items()},
                **{f"sc_var_{name}": value for name, value in sc_vars.items()},
                **{
                    f"ipc_{key}": value
                    for key, value in simulation_accounting.items()
                },
                **design,
            }
            results_log.append(row)
            eval_costs.append(obj_cost)
            eval_perfs.append(obj_perf)

            hv = _hypervolume_2d_min_strict(np.asarray(eval_costs), np.asarray(eval_perfs), ref_point)
            prev_hv = hv_log[-1]["hypervolume"] if hv_log else 0.0
            hv_log.append(
                {
                    "iter": attempts - 1,
                    "gidx": int(gidx),
                    "hypervolume": hv,
                    "delta_hv": hv - prev_hv,
                    "ref_cost": float(ref_point[0]),
                    "ref_perf": float(ref_point[1]),
                }
            )
            LOG.info(
                "Iter %d/%d | gidx %d | Cost: %.4f | Metric(%s)=%.4f | ObjPerf=%.4f",
                accepted + 1,
                budget,
                gidx,
                obj_cost,
                perf_metric,
                portfolio_metric,
                obj_perf,
            )
            accepted += 1
            _persist_campaign_state()
            gate_status = _operational_gate_status()
            _write_json_atomic(gate_path, gate_status)
            if gate_status["stop_required"]:
                raise RuntimeError(
                    "Operational gate stopped the campaign after a durable design "
                    f"checkpoint: {gate_status['breaches']}. Review infrastructure "
                    "before explicitly resuming the unchanged campaign."
                )
    finally:
        if sim is not None:
            sim.close()

    df_res = pd.DataFrame(results_log)
    _persist_campaign_state()

    if df_res.empty:
        LOG.warning("Optimization finished with no recorded evaluations. Wrote empty history/frontier in %s", out_dir)
        print("No optimization evaluations were recorded.")
        return

    costs = df_res["obj_cost"].values
    perfs = df_res["obj_perf"].values
    is_pareto = _non_dominated_mask(costs, perfs)
    df_pareto = df_res.iloc[is_pareto]

    LOG.info("Optimization Done. Found %d Pareto optimal designs. Artifacts in %s", len(df_pareto), out_dir)
    print(df_pareto[["gidx", "obj_cost", "obj_perf"]])
