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
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.fresh_seed_policy import FreshSeedPolicy
from source.design_exploration.simulator_interface.carla_worker_pool import CarlaWorkerPool
from source.design_exploration.simulator_interface.new_sim_interface import SimInterface
from source.design_exploration.simulator_interface.worker_protocol import (
    raise_worker_failure,
)
from source.simulation_environment.rare_events.seeds import derive_episode_sampler_seeds


class PooledSimInterface:
    """
    SimInterface-compatible adapter backed by CarlaWorkerPool.

    The adapter submits one worker-pool job per design and batches episode seeds
    within that job to reduce orchestration overhead.
    """

    def __init__(
        self,
        output_folder: str,
        algo_version: str,
        num_workers: Optional[int] = None,
        seed_base: int = 0,
        *,
        num_carla_instances: Optional[int] = None,
        carla_path: Optional[str] = None,
    ):
        workers = num_workers if num_workers is not None else num_carla_instances
        workers = 3 if workers is None else int(workers)
        self.output_folder = output_folder
        self.algo_version = algo_version
        self.num_carla_instances = workers
        self.carla_path = carla_path or SimInterface.CARLA_PATH
        if not os.path.isfile(self.carla_path) or not os.access(self.carla_path, os.X_OK):
            raise FileNotFoundError(
                "A CARLA executable is required (--carla-path or $CARLA_PATH): "
                f"{self.carla_path!r}"
            )
        self._seed_policy = FreshSeedPolicy(int(seed_base))
        self._used_seeds: Dict[str, set[int]] = defaultdict(set)
        self._pool = CarlaWorkerPool(
            num_workers=workers,
            algo_version=algo_version,
            output_root=output_folder,
            carla_path=self.carla_path,
        )

    @staticmethod
    def _to_episode_result(payload: Any) -> EpisodeResult:
        if isinstance(payload, EpisodeResult):
            return payload
        if hasattr(payload, "result"):
            return payload
        ep = EpisodeResult()
        ep.result = payload if isinstance(payload, dict) else {"payload": payload}
        return ep

    @staticmethod
    def _design_seed_key(design_dict: Dict[str, Any]) -> str:
        txt = json.dumps(design_dict, sort_keys=True, default=float)
        return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]

    def _next_unique_seed(self, design_seed_key: str, scenario_name: str) -> int:
        for _ in range(1000):
            seed_val = int(self._seed_policy.next_seed(design_seed_key, scenario_name))
            if seed_val not in self._used_seeds[design_seed_key]:
                self._used_seeds[design_seed_key].add(seed_val)
                return seed_val
        raise RuntimeError(f"Unable to allocate a fresh seed for key={design_seed_key} scenario={scenario_name}")

    def simulate(
        self,
        scenario_name: str,
        base_cfg: Dict[str, Any],
        design,
        num_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> List[EpisodeResult]:
        n = int(max(0, num_episodes))
        if n == 0:
            return []
        if seeds is not None and len(seeds) != n:
            raise ValueError(f"Expected {n} explicit seeds, got {len(seeds)}")

        design_dict = dict(design)
        design_seed_key = self._design_seed_key(design_dict)
        seed_list = (
            [int(seed) for seed in seeds]
            if seeds is not None
            else [self._next_unique_seed(design_seed_key, scenario_name) for _ in range(n)]
        )
        rare_event_cfg = base_cfg.get("rare_event", {})
        if rare_event_cfg.get("enabled", False):
            derive_episode_sampler_seeds(seed_list, rare_event_cfg.get("seed"))
        # Split a single design evaluation into deterministic contiguous chunks
        # so all configured CARLA workers are used while output order is stable.
        chunk_count = min(self.num_carla_instances, n)
        chunk_size = (n + chunk_count - 1) // chunk_count
        chunks = [seed_list[start:start + chunk_size] for start in range(0, n, chunk_size)]
        job_ids = [
            self._pool.submit(
                scenario_name=scenario_name,
                scenario_cfg=base_cfg,
                design=design_dict,
                seeds=chunk,
            )
            for chunk in chunks
        ]
        completed_payloads: Dict[str, List[Any]] = {}

        while len(completed_payloads) < len(job_ids):
            completed = self._pool.poll()
            if not completed:
                time.sleep(0.05)
                continue

            for rec in completed:
                if rec.job_id not in job_ids:
                    continue
                if rec.status != "done":
                    raise_worker_failure(
                        reason=rec.failure_reason,
                        error=rec.error,
                        job_id=rec.job_id,
                    )
                payload_list = rec.payload if isinstance(rec.payload, list) else [rec.payload]
                completed_payloads[rec.job_id] = payload_list

        ordered_payloads = [
            payload
            for job_id in job_ids
            for payload in completed_payloads[job_id]
        ]
        if len(ordered_payloads) != n:
            raise RuntimeError(f"Worker pool returned {len(ordered_payloads)} results for {n} seeds")
        return [self._to_episode_result(payload) for payload in ordered_payloads]

    def close(self) -> None:
        self._pool.shutdown()
