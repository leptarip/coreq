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

"""Offline tests for ParEGO campaign and IPC episode continuation."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from source.design_exploration.commons.design_space import SPACE_SPEC
from source.design_exploration.optimization.parego_qrf_common import ParEGOQuantileRF
from source.design_exploration.optimization.parego_qrf_ipc_client import (
    replay_optimizer_history,
)

from source.design_exploration.optimization.sim_ipc_server import PoolBackedSimServer


class _FakePool:
    fatal_error = None
    alive_workers = 3

    def __init__(self, *, fail_after=None, terminal_failure_seed=None):
        self.fail_after = fail_after
        self.terminal_failure_seed = terminal_failure_seed
        self.completed = 0
        self.counter = 0
        self.queue = []
        self.peak_queued = 0
        self.submitted_seeds = []

    def submit(self, **kwargs):
        job_id = f"job-{self.counter}"
        self.counter += 1
        self.queue.append((job_id, copy.deepcopy(kwargs)))
        self.submitted_seeds.extend(int(seed) for seed in kwargs["seeds"])
        self.peak_queued = max(self.peak_queued, len(self.queue))
        return job_id

    def poll(self):
        if self.fail_after is not None and self.completed >= self.fail_after:
            raise RuntimeError("synthetic server interruption")
        if not self.queue:
            return []
        job_id, kwargs = self.queue.pop(0)
        self.completed += 1
        if self.terminal_failure_seed in kwargs["seeds"]:
            return [
                SimpleNamespace(
                    job_id=job_id,
                    status="failed",
                    payload=None,
                    attempts=10,
                    error="synthetic CARLA death",
                )
            ]
        payloads = [
            {"kpi": {"vel_at_min_d": float(seed % 10)}}
            for seed in kwargs["seeds"]
        ]
        return [
            SimpleNamespace(
                job_id=job_id,
                status="done",
                payload=payloads,
                attempts=1,
                error=None,
            )
        ]


def _offline_server(tmp_path, pool):
    server = PoolBackedSimServer.__new__(PoolBackedSimServer)
    server.scenario_cfgs = {"intersection1": "", "intersection2": ""}
    server.helpers = {"intersection1": {}, "intersection2": {}}
    server.algo_version = "parego_qrf_ipc_v3"
    server.sim_out = str(tmp_path)
    server.num_workers = 3
    server.episode_chunk_size = 1
    server.max_in_flight = 2
    server.checkpoint_root = tmp_path / "ipc_request_checkpoints"
    server.checkpoint_root.mkdir(parents=True, exist_ok=True)
    server.pool = pool
    return server


def _request():
    grid = SPACE_SPEC.build_grid()
    design = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in zip(SPACE_SPEC.keys, grid[0])
    }
    return {
        "id": "test-request",
        "action": "evaluate_design",
        "design": design,
        "scenario_seeds": [
            {"name": "intersection1", "seeds": [101, 102, 103]},
            {"name": "intersection2", "seeds": [201, 202, 203]},
        ],
    }


def test_ipc_resumes_completed_single_episode_chunks_with_bounded_queue(tmp_path):
    interrupted_pool = _FakePool(fail_after=2)
    interrupted = _offline_server(tmp_path, interrupted_pool)
    first = interrupted._handle_evaluate_design(_request())
    assert first["ok"] is False
    assert interrupted_pool.peak_queued <= 2

    resumed_pool = _FakePool()
    resumed = _offline_server(tmp_path, resumed_pool)
    second = resumed._handle_evaluate_design(_request())

    assert second["ok"] is True
    assert second["accounting"] == {
        "planned_chunks": 6,
        "resumed_chunks": 2,
        "partially_resumed_chunks": 0,
        "resumed_episodes": 2,
        "submitted_chunks": 4,
        "submitted_episodes": 4,
        "submitted_worker_attempts": 4,
        "retried_chunks": 0,
        "resumed_recorded_attempts": 2,
        "prior_failed_worker_attempts": 0,
        "prior_terminal_failures": 0,
        "total_recorded_worker_attempts": 6,
        "episode_chunk_size": 1,
        "effective_chunk_size": 1,
        "max_in_flight": 2,
    }
    assert resumed_pool.peak_queued <= 2
    assert len(resumed_pool.submitted_seeds) == 4
    assert [item["seed"] for item in second["episodes"]["intersection1"]] == [
        101,
        102,
        103,
    ]
    assert [item["seed"] for item in second["episodes"]["intersection2"]] == [
        201,
        202,
        203,
    ]


def test_ipc_carries_terminal_retry_attempts_into_resumed_accounting(tmp_path):
    failed_pool = _FakePool(terminal_failure_seed=103)
    failed = _offline_server(tmp_path, failed_pool)
    first = failed._handle_evaluate_design(_request())
    assert first["ok"] is False

    resumed = _offline_server(tmp_path, _FakePool())._handle_evaluate_design(_request())
    assert resumed["ok"] is True
    # Three chunks survive the interrupted request, not two: the job that was
    # already in flight when seed 103 died is now drained and checkpointed
    # instead of being abandoned with the request.
    assert resumed["accounting"]["resumed_chunks"] == 3
    assert resumed["accounting"]["submitted_chunks"] == 3
    assert resumed["accounting"]["prior_failed_worker_attempts"] == 10
    assert resumed["accounting"]["prior_terminal_failures"] == 1
    assert resumed["accounting"]["total_recorded_worker_attempts"] == 16


def test_optimizer_prefix_replay_restores_next_proposal_exactly():
    pytest.importorskip("quantile_forest")
    grid = SPACE_SPEC.build_grid()
    safe = np.arange(40, dtype=int)
    lo = grid.min(axis=0)
    span = np.where(grid.max(axis=0) > lo, grid.max(axis=0) - lo, 1.0)
    x_safe = ((grid - lo) / span)[safe]

    original = ParEGOQuantileRF(x_safe, safe, random_seed=11)
    rows = []
    for position in range(8):
        gidx = original.ask()
        rows.append(
            {
                "iter": position,
                "gidx": gidx,
                "obj_cost": 1000.0 + position,
                "obj_perf": -1.0 - position / 100.0,
                "obs_var": 0.01,
                "profile": original.profile,
                "acquisition": original.acquisition,
                "acq_value": original.last_acq,
                "scalarization_weights": (
                    None
                    if original.last_weights is None
                    else original.last_weights.tolist()
                ),
            }
        )
        original.tell(gidx, [1000.0 + position, -1.0 - position / 100.0], 0.01)

    restored = ParEGOQuantileRF(x_safe, safe, random_seed=11)
    state = replay_optimizer_history(restored, rows)
    assert state["attempts"] == 8
    assert state["accepted"] == 8
    assert restored.ask() == original.ask()


def test_optimizer_prefix_replay_rejects_design_drift():
    pytest.importorskip("quantile_forest")
    grid = SPACE_SPEC.build_grid()
    safe = np.arange(20, dtype=int)
    lo = grid.min(axis=0)
    span = np.where(grid.max(axis=0) > lo, grid.max(axis=0) - lo, 1.0)
    optimizer = ParEGOQuantileRF(
        ((grid - lo) / span)[safe], safe, random_seed=11
    )
    with pytest.raises(RuntimeError, match="Deterministic continuation check failed"):
        replay_optimizer_history(
            optimizer,
            [
                {
                    "iter": 0,
                    "gidx": 999999,
                    "obj_cost": 1.0,
                    "obj_perf": -1.0,
                }
            ],
        )


class _BatchPool:
    """Completes every queued job in one poll(), failure first.

    `_FakePool` returns one record per poll, so it can never produce a batch
    holding a failure *and* successes. A real `CarlaWorkerPool.poll()` drains its
    whole result queue, which is the case that used to discard completed work.
    """

    fatal_error = None
    alive_workers = 3

    def __init__(self, failing_seed):
        self.failing_seed = failing_seed
        self.counter = 0
        self.queue = []
        self.submitted_seeds = []

    def submit(self, **kwargs):
        job_id = f"job-{self.counter}"
        self.counter += 1
        self.queue.append((job_id, copy.deepcopy(kwargs)))
        self.submitted_seeds.extend(int(seed) for seed in kwargs["seeds"])
        return job_id

    def poll(self):
        if not self.queue:
            return []
        batch, self.queue = self.queue, []
        records = []
        for job_id, kwargs in batch:
            if self.failing_seed in kwargs["seeds"]:
                records.append(SimpleNamespace(
                    job_id=job_id, status="failed", payload=None,
                    attempts=10, error="synthetic CARLA death"))
            else:
                records.append(SimpleNamespace(
                    job_id=job_id, status="done", attempts=1, error=None,
                    payload=[{"kpi": {"vel_at_min_d": float(seed % 10)}}
                             for seed in kwargs["seeds"]]))
        records.sort(key=lambda record: record.status != "failed")
        return records


def _stored_episodes(server):
    return sorted(
        path for path in server.checkpoint_root.rglob("*.json")
        if not path.name.endswith(".failures.json")
    )


def test_ipc_checkpoints_siblings_of_a_failed_job_in_the_same_batch(tmp_path):
    """A CARLA death must not discard the episodes that completed beside it."""
    server = _offline_server(tmp_path, _BatchPool(failing_seed=101))
    server.max_in_flight = 6

    response = server._handle_evaluate_design(_request())

    assert response["ok"] is False
    assert len(_stored_episodes(server)) == 5

    resumed = _offline_server(tmp_path, _FakePool())
    resumed.max_in_flight = 6
    second = resumed._handle_evaluate_design(_request())
    assert second["ok"] is True
    assert second["accounting"]["resumed_chunks"] == 5
    assert second["accounting"]["submitted_chunks"] == 1


def test_ipc_checkpoints_survive_a_changed_episode_chunk_size(tmp_path):
    """Episodes are keyed individually, so re-chunking must not re-simulate."""
    first = _offline_server(tmp_path, _FakePool())
    first.max_in_flight = 6
    assert first._handle_evaluate_design(_request())["ok"] is True

    pool = _FakePool()
    resumed = _offline_server(tmp_path, pool)
    resumed.episode_chunk_size = 3
    resumed.max_in_flight = 6
    second = resumed._handle_evaluate_design(_request())

    assert second["ok"] is True
    assert second["accounting"]["submitted_chunks"] == 0
    assert pool.submitted_seeds == []
    assert [item["seed"] for item in second["episodes"]["intersection1"]] == [101, 102, 103]


def test_ipc_refuses_checkpoints_written_under_another_algo_version(tmp_path):
    """Resuming must not silently mix episodes from a different simulator build."""
    first = _offline_server(tmp_path, _FakePool())
    first.max_in_flight = 6
    assert first._handle_evaluate_design(_request())["ok"] is True

    other = _offline_server(tmp_path, _FakePool())
    other.max_in_flight = 6
    other.algo_version = "parego_qrf_ipc_other"
    second = other._handle_evaluate_design(_request())

    assert second["ok"] is True
    assert second["accounting"]["resumed_chunks"] == 0
    assert second["accounting"]["submitted_chunks"] == 6


def test_ipc_failure_journal_isolated_by_algorithm_and_scenario_config(tmp_path):
    failed = _offline_server(tmp_path, _FakePool(terminal_failure_seed=101))
    failed.max_in_flight = 6
    assert failed._handle_evaluate_design(_request())["ok"] is False

    other = _offline_server(tmp_path, _FakePool())
    other.max_in_flight = 6
    other.algo_version = "parego_qrf_ipc_other"
    response = other._handle_evaluate_design(_request())

    assert response["ok"] is True
    assert response["accounting"]["prior_failed_worker_attempts"] == 0
    assert response["accounting"]["prior_terminal_failures"] == 0


def test_ipc_failure_accounting_survives_rechunking(tmp_path):
    failed = _offline_server(tmp_path, _FakePool(terminal_failure_seed=101))
    failed.max_in_flight = 6
    assert failed._handle_evaluate_design(_request())["ok"] is False

    resumed = _offline_server(tmp_path, _FakePool())
    resumed.episode_chunk_size = 3
    resumed.max_in_flight = 6
    response = resumed._handle_evaluate_design(_request())

    assert response["ok"] is True
    assert response["accounting"]["prior_failed_worker_attempts"] == 10
    assert response["accounting"]["prior_terminal_failures"] == 1


def test_ipc_keeps_every_worker_busy_at_large_chunk_sizes(tmp_path):
    """A chunk size above the per-worker share must not idle a worker."""
    server = _offline_server(tmp_path, _FakePool())
    server.episode_chunk_size = 20
    plans = [
        {"name": "intersection1", "cfg_base": {}, "seeds": list(range(20))},
        {"name": "intersection2", "cfg_base": {}, "seeds": list(range(100, 120))},
    ]

    chunks = server._plan_chunks(plans)

    assert len(chunks) >= server.num_workers
    assert max(len(chunk["seeds"]) for chunk in chunks) <= 20
    assert sum(len(chunk["seeds"]) for chunk in chunks) == 40


def test_ipc_accounting_reports_the_chunk_size_actually_used(tmp_path):
    """Reporting only the configured size would misdescribe a rebalanced run."""
    server = _offline_server(tmp_path, _FakePool())
    server.episode_chunk_size = 3
    server.max_in_flight = 6

    accounting = server._handle_evaluate_design(_request())["accounting"]

    assert accounting["episode_chunk_size"] == 3
    assert accounting["effective_chunk_size"] == 2
    # Splitting is per scenario, so two scenarios of three seeds at an effective
    # size of two give four chunks -- comfortably above the three workers.
    assert accounting["planned_chunks"] == 4


class _FatalAfterPollPool(_BatchPool):
    """Expose completed records and a fatal pool state in the same poll."""

    def __init__(self):
        super().__init__(failing_seed=-1)
        self.fatal_error = None
        self.alive_workers = 3

    def poll(self):
        records = super().poll()
        if records:
            self.fatal_error = "synthetic pool fatal state"
            self.alive_workers = 0
        return records


def test_ipc_drains_completed_records_when_poll_also_sets_fatal_error(tmp_path):
    server = _offline_server(tmp_path, _FatalAfterPollPool())
    server.max_in_flight = 6

    response = server._handle_evaluate_design(_request())

    assert response["ok"] is False
    assert "synthetic pool fatal state" in response["error"]
    assert len(_stored_episodes(server)) == 6


def test_ipc_restores_a_scenario_from_a_narrower_retry_request(tmp_path):
    first = _offline_server(tmp_path, _FakePool())
    first.max_in_flight = 6
    assert first._handle_evaluate_design(_request())["ok"] is True

    narrower = _request()
    narrower["id"] = "missing-scenario-retry"
    narrower["scenario_seeds"] = [narrower["scenario_seeds"][1]]
    pool = _FakePool()
    resumed = _offline_server(tmp_path, pool)
    resumed.max_in_flight = 6

    response = resumed._handle_evaluate_design(narrower)

    assert response["ok"] is True
    assert response["accounting"]["resumed_episodes"] == 3
    assert response["accounting"]["submitted_episodes"] == 0
    assert pool.submitted_seeds == []


def test_ipc_submits_only_missing_episodes_from_a_partial_chunk(tmp_path):
    first = _offline_server(tmp_path, _FakePool())
    first.max_in_flight = 6
    assert first._handle_evaluate_design(_request())["ok"] is True
    for path in _stored_episodes(first):
        record = json.loads(path.read_text())
        if int(record["seed"]) != 101:
            path.unlink()

    pool = _FakePool()
    resumed = _offline_server(tmp_path, pool)
    resumed.episode_chunk_size = 3
    resumed.max_in_flight = 6
    response = resumed._handle_evaluate_design(_request())

    assert response["ok"] is True
    assert response["accounting"]["partially_resumed_chunks"] == 1
    assert response["accounting"]["resumed_episodes"] == 1
    assert response["accounting"]["submitted_episodes"] == 5
    assert 101 not in pool.submitted_seeds


def test_resumed_attempts_are_counted_once_per_originating_worker_job(tmp_path):
    first = _offline_server(tmp_path, _FakePool())
    first.episode_chunk_size = 3
    first.max_in_flight = 6
    first_response = first._handle_evaluate_design(_request())
    assert first_response["accounting"]["planned_chunks"] == 4

    resumed = _offline_server(tmp_path, _FakePool())
    resumed.episode_chunk_size = 1
    resumed.max_in_flight = 6
    response = resumed._handle_evaluate_design(_request())

    assert response["ok"] is True
    assert response["accounting"]["resumed_chunks"] == 6
    assert response["accounting"]["resumed_recorded_attempts"] == 4
    assert response["accounting"]["total_recorded_worker_attempts"] == 4

