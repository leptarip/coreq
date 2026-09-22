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
Minimal per-port CARLA worker process.

Each worker:
- launches a dedicated CARLA instance on a fixed port,
- pulls jobs from a shared task queue,
- runs `_simulate_single` directly (no ProcessPoolExecutor),
- pushes results or failures to a result queue,
- restarts CARLA locally on crash without affecting other workers.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import queue
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

from source.design_exploration.commons.design_space import Design
from source.design_exploration.simulator_interface.new_sim_interface import (
    SimInterface,
    _simulate_single,
    _start_carla_stderr_drain,
)
from source.design_exploration.simulator_interface.worker_protocol import (
    attach_verified_seed_plan,
    failure_reason_for_exception,
)

logger = logging.getLogger(__name__)


def _publish_carla_ownership(cfg: "WorkerConfig", proc: subprocess.Popen) -> None:
    """Persist the exact CARLA process group owned by this worker."""
    root = os.path.join(os.path.abspath(cfg.output_root), "carla_ownership")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"worker_{int(cfg.worker_id)}.json")
    temp = f"{path}.{os.getpid()}.tmp"
    record = {
        "schema_version": 1,
        "worker_id": int(cfg.worker_id),
        "worker_pid": int(os.getpid()),
        "port": int(cfg.port),
        "carla_path": os.path.realpath(cfg.carla_path),
        "launcher_pid": int(proc.pid),
        "carla_pgid": int(getattr(proc, "_carla_pgid", proc.pid)),
        "started_unix": time.time(),
    }
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@dataclass
class WorkerConfig:
    worker_id: int
    port: int
    carla_path: str
    output_root: str
    algo_version: str
    heartbeat_s: float = 5.0
    restart_backoff_s: float = 3.0
    startup_delay_s: float = 0.0
    startup_retries: int = 3
    startup_retry_backoff_s: float = 5.0


def _start_carla(
    carla_path: str,
    port: int,
    *,
    process_started: Optional[Callable[[subprocess.Popen], None]] = None,
) -> subprocess.Popen:
    """
    Start a single CARLA instance bound to the given port.
    Returns the running subprocess; caller owns termination.
    """
    carla_port = f"-carla-port={port}"
    streaming_port = f"-carla-streaming-port={port + 1}"
    logger.info("Launching CARLA on port %d", port)
    p_carla = subprocess.Popen(
        [carla_path, "-RenderOffScreen", "-nosound", carla_port, streaming_port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    # preexec_fn=os.setsid makes the launcher's PID its process-group ID. Keep
    # it even if the wrapper exits before cleanup, because the shipping binary
    # may still be alive in that group.
    p_carla._carla_pgid = int(p_carla.pid)  # type: ignore[attr-defined]
    if process_started is not None:
        # Publish ownership before the startup sleep so a signal during that
        # window can still terminate this exact CARLA process group.
        process_started(p_carla)
    time.sleep(3)
    ret_code = p_carla.poll()
    if ret_code is not None:
        stderr_line = p_carla.stderr.readline()
        logger.error("CARLA failed to start on port %d (code %s): %s", port, ret_code, stderr_line)
        # Kill the entire process group to clean up any child processes (e.g. shader
        # compilers) that survived the parent's crash.  Using the pgid is precise:
        # it targets exactly this launch, not any other CARLA instance on the host.
        try:
            pgid = os.getpgid(p_carla.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            p_carla.wait(timeout=5)
        except Exception:
            pass
        raise RuntimeError(f"CARLA failed to start on port {port}")
    _start_carla_stderr_drain(p_carla, port)
    return p_carla


def _terminate_carla(p: Optional[subprocess.Popen], port: int) -> None:
    if not p:
        return
    pgid = getattr(p, "_carla_pgid", None)
    if pgid is None and p.poll() is None:
        try:
            pgid = os.getpgid(p.pid)
        except (ProcessLookupError, OSError):
            pgid = None
    try:
        if pgid is not None:
            os.killpg(int(pgid), signal.SIGTERM)
            time.sleep(2)
            os.killpg(int(pgid), signal.SIGKILL)
        elif p.poll() is None:
            p.terminate()
    except ProcessLookupError:
        logger.info("CARLA process for port %d already gone.", port)
    except Exception as exc:
        logger.warning("Error terminating CARLA on port %d: %s", port, exc)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Timed out reaping CARLA launcher for port %d", port)


def _ensure_output_dir(base: str, port: int) -> str:
    out_dir = os.path.join(os.path.abspath(base), f"port_{port}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _next_counter_from_disk(out_dir: str, start: int = 0) -> int:
    """
    Avoid directory collisions after worker restarts by resuming the counter
    from the largest existing numeric subfolder under out_dir.
    """
    try:
        existing = [
            int(name) for name in os.listdir(out_dir) if name.isdigit()
        ]
        return max(existing, default=start - 1) + 1
    except Exception:
        # Fallback: just continue from provided start
        return start


def _start_carla_with_retries(
    cfg: WorkerConfig,
    *,
    initial_delay_s: float,
    process_started: Optional[Callable[[subprocess.Popen], None]] = None,
) -> subprocess.Popen:
    if initial_delay_s > 0:
        logger.info(
            "Worker %d delaying CARLA startup on port %d by %.1fs",
            cfg.worker_id,
            cfg.port,
            initial_delay_s,
        )
        time.sleep(initial_delay_s)

    last_exc: Optional[Exception] = None
    for attempt in range(int(cfg.startup_retries)):
        try:
            return _start_carla(
                cfg.carla_path,
                cfg.port,
                process_started=process_started,
            )
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= int(cfg.startup_retries):
                break
            backoff = float(cfg.startup_retry_backoff_s) * float(attempt + 1)
            logger.warning(
                "Worker %d failed to start CARLA on port %d (attempt %d/%d); retrying in %.1fs",
                cfg.worker_id,
                cfg.port,
                attempt + 1,
                cfg.startup_retries,
                backoff,
            )
            time.sleep(backoff)
    if last_exc is None:
        raise RuntimeError(f"CARLA failed to start on port {cfg.port}")
    raise last_exc


def worker_main(task_queue, result_queue, cfg: WorkerConfig):
    """
    Entry point for a worker process.
    Messages sent to result_queue are dictionaries:
      {"type": "heartbeat", "worker_id": int, "ts": float}
      {"type": "started", "worker_id": int, "job_id": str, "ts": float}
      {"type": "result", "status": "ok"|"failed", "worker_id": int, "job_id": str, "error": str|None, "payload": Any}
    """
    carla_proc: Optional[subprocess.Popen] = None
    out_dir = _ensure_output_dir(cfg.output_root, cfg.port)
    counter = _next_counter_from_disk(out_dir, start=0)
    active_job_id: Optional[str] = None
    previous_signal_handlers = {}

    def _terminate_cleanly(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + int(signum))

    def _record_carla_process(proc: subprocess.Popen) -> None:
        nonlocal carla_proc
        carla_proc = proc
        _publish_carla_ownership(cfg, proc)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _terminate_cleanly)
    try:
        carla_proc = _start_carla_with_retries(
            cfg,
            initial_delay_s=float(cfg.startup_delay_s),
            process_started=_record_carla_process,
        )
        while True:
            try:
                job = task_queue.get(timeout=cfg.heartbeat_s)
            except queue.Empty:
                result_queue.put(
                    {"type": "heartbeat", "worker_id": cfg.worker_id, "ts": time.time(), "active_job_id": active_job_id}
                )
                continue

            if job is None:
                break  # shutdown signal

            job_id = job.get("job_id")
            scenario_name = job["scenario_name"]
            scenario_cfg = copy.deepcopy(job["scenario_cfg"])
            design = Design(job["design"])
            seed_list_raw = job.get("seeds")
            seed_list = None if seed_list_raw is None else [int(s) for s in seed_list_raw]
            num_episodes = len(seed_list) if seed_list is not None else int(job.get("num_episodes", 1))
            seed = job.get("seed", None)
            scenario_cfg["sim"] = dict(scenario_cfg.get("sim", {}))
            if seed_list is not None:
                scenario_cfg["sim"]["seed_list"] = list(seed_list)
                scenario_cfg["sim"].pop("seed", None)
            elif seed is not None:
                scenario_cfg["sim"] = dict(scenario_cfg.get("sim", {}))
                scenario_cfg["sim"]["seed"] = int(seed)
                scenario_cfg["sim"].pop("seed_list", None)
            else:
                scenario_cfg["sim"].pop("seed", None)
                scenario_cfg["sim"].pop("seed_list", None)

            result_queue.put(
                {"type": "started", "worker_id": cfg.worker_id, "job_id": job_id, "ts": time.time()}
            )
            active_job_id = job_id
            counter += 1
            try:
                payload = _simulate_single(
                    scenario_cfg,
                    SimInterface.SERVER_IP,
                    cfg.port,
                    out_dir,
                    counter,
                    scenario_name,
                    design,
                    num_episodes,
                    cfg.algo_version,
                )
                if not payload:
                    raise RuntimeError("Empty payload from simulator")
                payload = attach_verified_seed_plan(
                    payload,
                    seed=seed,
                    seeds=seed_list,
                    num_episodes=num_episodes,
                )
                result_queue.put(
                    {
                        "type": "result",
                        "status": "ok",
                        "worker_id": cfg.worker_id,
                        "job_id": job_id,
                        "payload": payload,
                        "error": None,
                    }
                )
                active_job_id = None
            except SimInterface.CarlaServerDead as exc:
                result_queue.put(
                    {
                        "type": "result",
                        "status": "failed",
                        "reason": "carla_dead",
                        "worker_id": cfg.worker_id,
                        "job_id": job_id,
                        "payload": None,
                        "error": str(exc),
                    }
                )
                active_job_id = None
                _terminate_carla(carla_proc, cfg.port)
                carla_proc = _start_carla_with_retries(
                    cfg,
                    initial_delay_s=float(cfg.restart_backoff_s),
                    process_started=_record_carla_process,
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "result",
                        "status": "failed",
                        "reason": failure_reason_for_exception(exc),
                        "worker_id": cfg.worker_id,
                        "job_id": job_id,
                        "payload": None,
                        "error": str(exc),
                    }
                )
                active_job_id = None
    finally:
        _terminate_carla(carla_proc, cfg.port)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
