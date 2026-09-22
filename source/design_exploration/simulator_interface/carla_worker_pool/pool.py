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
Lightweight orchestrator for CARLA worker processes.

Usage:
    pool = CarlaWorkerPool(num_workers=3, algo_version="prob_eval")
    jid = pool.submit(scenario_name, scenario_cfg, design_dict, seed=123)
    while pool.pending:
        completed = pool.poll()
    pool.shutdown()
"""
from __future__ import annotations

import copy
import logging
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import multiprocessing as mp

from source.design_exploration.simulator_interface.new_sim_interface import SimInterface
from source.design_exploration.simulator_interface.carla_worker_pool.worker import (
    WorkerConfig,
    worker_main,
)
from source.design_exploration.simulator_interface.worker_protocol import (
    CONFIGURATION_ERROR_REASON,
    attach_verified_seed_plan,
    requested_episode_seeds,
    should_retry_failure,
)
from source.simulation_environment.configuration_error import SimulationConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    scenario_name: str
    scenario_cfg: Dict[str, Any]
    design: Dict[str, Any]
    seed: Optional[int]  # immutable requested base seed
    seeds: Optional[List[int]]  # immutable requested per-episode seed plan
    num_episodes: int
    attempts: int = 0
    max_retries: int = 1
    status: str = "pending"  # pending|running|done|failed
    running_since: Optional[float] = None
    payload: Any = None
    error: Optional[str] = None
    failure_reason: Optional[str] = None
    worker_id: Optional[int] = None


class CarlaWorkerPool:
    def __init__(
        self,
        num_workers: int,
        *,
        base_port: int = SimInterface.BASE_PORT,
        carla_path: str = SimInterface.CARLA_PATH,
        output_root: str = "carla_worker_runs",
        algo_version: str = "prob_eval",
        heartbeat_s: float = 5.0,
        max_retries: int = 10,
        spawn_stagger_s: float = 6.0,
        max_worker_respawns: int = 2,
    ):
        self.num_workers = int(num_workers)
        self.ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue = self.ctx.Queue()
        self.result_queue: mp.Queue = self.ctx.Queue()
        self.output_root = output_root
        self.carla_path = carla_path
        self.algo_version = algo_version
        self.heartbeat_s = heartbeat_s
        self.max_retries_default = max_retries
        self.spawn_stagger_s = float(spawn_stagger_s)
        self.max_worker_respawns = int(max_worker_respawns)

        self._workers: List[mp.Process] = []
        self._worker_cfgs: List[WorkerConfig] = []
        self._heartbeats: Dict[int, float] = {}
        self._jobs: Dict[str, JobRecord] = {}
        self._worker_respawns: Dict[int, int] = {}
        self._disabled_worker_ids: set[int] = set()
        self._fatal_error: Optional[str] = None

        for idx in range(self.num_workers):
            port = base_port + idx * 1000
            cfg = WorkerConfig(
                worker_id=idx,
                port=port,
                carla_path=self.carla_path,
                output_root=self.output_root,
                algo_version=self.algo_version,
                heartbeat_s=self.heartbeat_s,
                startup_delay_s=float(idx) * self.spawn_stagger_s,
            )
            self._worker_cfgs.append(cfg)
            proc = self._spawn_worker(cfg)
            self._workers.append(proc)

    # ---------- public API ----------
    def submit(
        self,
        scenario_name: str,
        scenario_cfg: Dict[str, Any],
        design: Dict[str, Any],
        *,
        seed: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        num_episodes: int = 1,
        job_id: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        seed_list = None if seeds is None else [int(s) for s in seeds]
        episodes = len(seed_list) if seed_list is not None else int(num_episodes)
        requested_episode_seeds(
            seed=seed,
            seeds=seed_list,
            num_episodes=episodes,
        )
        jid = job_id or str(uuid.uuid4())
        rec = JobRecord(
            job_id=jid,
            scenario_name=scenario_name,
            scenario_cfg=copy.deepcopy(scenario_cfg),
            design=copy.deepcopy(design),
            seed=None if seed is None else int(seed),
            seeds=seed_list,
            num_episodes=int(episodes),
            max_retries=self.max_retries_default if max_retries is None else int(max_retries),
        )
        self._jobs[jid] = rec
        self.task_queue.put(
            dict(
                job_id=jid,
                scenario_name=scenario_name,
                scenario_cfg=rec.scenario_cfg,
                design=rec.design,
                seed=rec.seed,
                seeds=copy.deepcopy(rec.seeds),
                num_episodes=rec.num_episodes,
            )
        )
        return jid

    def poll(self) -> List[JobRecord]:
        """
        Drain result queue; returns JobRecords that reached a terminal state (done or failed with no retries left).
        """
        finished: List[JobRecord] = []
        while True:
            try:
                msg = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if msg.get("type") == "heartbeat":
                worker_id = msg["worker_id"]
                self._heartbeats[worker_id] = msg["ts"]
                self._worker_respawns[worker_id] = 0
                continue

            if msg.get("type") == "started":
                jid = msg.get("job_id")
                rec = self._jobs.get(jid)
                if rec is None:
                    logger.warning("Received started for unknown job_id=%s", jid)
                    continue
                self._worker_respawns[msg.get("worker_id")] = 0
                rec.status = "running"
                rec.running_since = msg.get("ts")
                rec.worker_id = msg.get("worker_id")
                continue

            if msg.get("type") != "result":
                continue

            jid = msg.get("job_id")
            rec = self._jobs.get(jid)
            if rec is None:
                logger.warning("Received result for unknown job_id=%s", jid)
                continue

            rec.worker_id = msg.get("worker_id")
            rec.error = msg.get("error")
            rec.failure_reason = msg.get("reason")
            rec.payload = msg.get("payload")
            rec.attempts += 1
            rec.running_since = None
            if rec.worker_id is not None:
                self._worker_respawns[rec.worker_id] = 0

            if msg.get("status") == "ok":
                try:
                    rec.payload = attach_verified_seed_plan(
                        rec.payload,
                        seed=rec.seed,
                        seeds=rec.seeds,
                        num_episodes=rec.num_episodes,
                    )
                except SimulationConfigurationError as exc:
                    rec.status = "failed"
                    rec.error = str(exc)
                    rec.failure_reason = CONFIGURATION_ERROR_REASON
                    finished.append(rec)
                    continue
                rec.status = "done"
                finished.append(rec)
            else:
                rec.status = "failed"
                if should_retry_failure(
                    rec.failure_reason,
                    attempts=rec.attempts,
                    max_retries=rec.max_retries,
                ):
                    logger.info("Retrying job %s (attempt %d/%d)", rec.job_id, rec.attempts, rec.max_retries)
                    self._enqueue_retry(rec)
                else:
                    finished.append(rec)

        finished.extend(self._restart_dead_workers())
        return finished

    @property
    def pending(self) -> int:
        return sum(1 for r in self._jobs.values() if r.status not in ("done", "failed"))

    def shutdown(self) -> None:
        for _ in self._workers:
            self.task_queue.put(None)
        for p in self._workers:
            p.join(timeout=10)
        for p in self._workers:
            if not p.is_alive():
                continue
            logger.warning("Worker PID %s did not exit cleanly; terminating", p.pid)
            try:
                p.terminate()
                p.join(timeout=5)
            except Exception:
                pass
        for p in self._workers:
            if not p.is_alive():
                continue
            logger.warning("Worker PID %s survived terminate(); killing", p.pid)
            try:
                p.kill()
                p.join(timeout=5)
            except Exception:
                pass
        try:
            self.task_queue.close()
            self.task_queue.join_thread()
        except Exception:
            pass
        try:
            self.result_queue.close()
            self.result_queue.join_thread()
        except Exception:
            pass
        self._workers.clear()

    @property
    def alive_workers(self) -> int:
        return sum(1 for p in self._workers if p.is_alive())

    @property
    def fatal_error(self) -> Optional[str]:
        return self._fatal_error

    # ---------- internals ----------
    def _enqueue_retry(self, rec: JobRecord) -> None:
        # A pool retry repeats the same logical job. Choosing a replacement seed
        # is an experiment-level decision and must be submitted as a new job.
        seed_retry = None if rec.seed is None else int(rec.seed)
        seeds_retry = None if rec.seeds is None else [int(seed) for seed in rec.seeds]
        self.task_queue.put(
            dict(
                job_id=rec.job_id,
                scenario_name=rec.scenario_name,
                scenario_cfg=rec.scenario_cfg,
                design=rec.design,
                seed=seed_retry,
                seeds=copy.deepcopy(seeds_retry),
                num_episodes=rec.num_episodes,
            )
        )
        rec.status = "pending"

    def _spawn_worker(self, cfg: WorkerConfig) -> mp.Process:
        proc = self.ctx.Process(
            target=worker_main,
            args=(self.task_queue, self.result_queue, cfg),
            daemon=True,
        )
        proc.start()
        self._heartbeats[cfg.worker_id] = time.time()
        return proc

    def _restart_dead_workers(self) -> List[JobRecord]:
        finished: List[JobRecord] = []
        new_procs = []
        for proc, cfg in zip(self._workers, self._worker_cfgs):
            if cfg.worker_id in self._disabled_worker_ids:
                # Keep the process/config lists aligned, but do not repeatedly
                # rediscover and report a worker whose respawn budget is gone.
                new_procs.append(proc)
                continue
            if proc.is_alive():
                new_procs.append(proc)
                continue
            respawns = self._worker_respawns.get(cfg.worker_id, 0) + 1
            self._worker_respawns[cfg.worker_id] = respawns
            lost_jobs = [j for j in self._jobs.values() if j.status == "running" and j.worker_id == cfg.worker_id]
            for rec in lost_jobs:
                rec.running_since = None
                rec.worker_id = None
                rec.attempts += 1
                if rec.attempts <= rec.max_retries:
                    logger.info("Requeuing job %s after worker %d death (attempt %d/%d)", rec.job_id, cfg.worker_id, rec.attempts, rec.max_retries)
                    self._enqueue_retry(rec)
                else:
                    rec.status = "failed"
                    rec.error = "worker_died"
                    finished.append(rec)
            if respawns > self.max_worker_respawns:
                msg = (
                    f"Worker {cfg.worker_id} on port {cfg.port} exceeded "
                    f"{self.max_worker_respawns} respawns and was disabled"
                )
                logger.error(msg)
                if self._fatal_error is None:
                    self._fatal_error = msg
                self._disabled_worker_ids.add(cfg.worker_id)
                new_procs.append(proc)
                continue
            logger.warning(
                "Worker %d died; restarting on port %d (respawn %d/%d)",
                cfg.worker_id,
                cfg.port,
                respawns,
                self.max_worker_respawns,
            )
            new_proc = self._spawn_worker(cfg)
            new_procs.append(new_proc)
        self._workers = new_procs
        return finished


__all__ = ["CarlaWorkerPool", "JobRecord"]
