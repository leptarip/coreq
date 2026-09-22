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

"""Process management of the ParEGO client, the Python 3.8 simulator server and its CARLA workers."""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

from source.design_exploration.commons.design_space import SPACE_SPEC
from source.design_exploration.optimization.parego_qrf_ipc_client import (
    IPCScenarioSpec,
    NDJSONProcessClient,
    run_parego_over_candidates,
)

COST_PROFILES = Path(__file__).resolve().parents[2] / "configuration" / "cost_profiles.json"


def _carla_worker_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(sys.executable).parents[1]))
    pytest.importorskip(
        "carla",
        reason="worker lifecycle tests require the Python 3.8 CARLA environment",
    )
    from source.design_exploration.simulator_interface.carla_worker_pool import worker

    return worker


def test_ipc_client_startup_timeout_reaps_its_server_group(tmp_path):
    server_script = tmp_path / "never_ready_server.py"
    server_pid_path = tmp_path / "server.pid"
    server_script.write_text(
        """\
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Timed out waiting"):
        NDJSONProcessClient(
            [sys.executable, str(server_script), str(server_pid_path)],
            ready_timeout=0.2,
        )

    server_pid = int(server_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)


def test_optimizer_client_passes_carla_path_to_python38_server(tmp_path, monkeypatch):
    pytest.importorskip("quantile_forest")
    observed = {}

    class StopBeforeServerLaunch(RuntimeError):
        pass

    def capture_server(command, worker_count):
        observed["command"] = list(command)
        observed["worker_count"] = worker_count
        raise StopBeforeServerLaunch

    monkeypatch.setattr(
        "source.design_exploration.optimization.parego_qrf_ipc_client.RemoteSimulator",
        capture_server,
    )
    carla_path = "/opt/carla/CarlaUE4.sh"
    with pytest.raises(StopBeforeServerLaunch):
        run_parego_over_candidates(
            space_spec=SPACE_SPEC,
            scenario_specs=[
                IPCScenarioSpec(
                    name="intersection1",
                    cfg_path="scenario.toml",
                    episodes_root=str(tmp_path / "episodes"),
                )
            ],
            python38=sys.executable,
            carla_path=carla_path,
            candidate_indices=np.arange(20),
            out_dir=str(tmp_path / "optimizer"),
            budget=20,
            num_carla_instances=3,
            cost_profile_config=str(COST_PROFILES),
            cost_profile_id="central",
            hv_ref_point=[274.0, 0.0],
            episode_seed_offset=400_000_000,
        )

    flag_index = observed["command"].index("--carla-path")
    assert observed["command"][flag_index + 1] == carla_path
    assert observed["worker_count"] == 3
    run_config = json.loads(
        (tmp_path / "optimizer" / "parego_qrf_ipc_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["carla_path"] == carla_path


def test_python38_server_passes_carla_path_to_worker_pool(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(sys.executable).parents[1]))
    from source.design_exploration.optimization import sim_ipc_server

    observed = {}

    class FakePool:
        alive_workers = 1

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def shutdown(self):
            observed["shutdown"] = True

    class InterruptedInput:
        def __iter__(self):
            return self

        def __next__(self):
            raise SystemExit(143)

    monkeypatch.setattr(sim_ipc_server, "CarlaWorkerPool", FakePool)
    monkeypatch.setattr(
        sim_ipc_server,
        "cfg_parser",
        type("CfgParser", (), {"parse": staticmethod(lambda _path: {})}),
    )
    carla_path = "/opt/carla/CarlaUE4.sh"
    server = sim_ipc_server.PoolBackedSimServer(
        {"intersection1": "scenario.toml"},
        "parego_qrf_ipc_v3",
        str(tmp_path / "sim"),
        3,
        carla_path,
    )
    monkeypatch.setattr(sys, "stdin", InterruptedInput())
    with pytest.raises(SystemExit, match="143"):
        server.serve()

    assert observed["carla_path"] == carla_path
    assert observed["num_workers"] == 3
    assert observed["shutdown"] is True


def test_worker_sigterm_runs_carla_cleanup_before_exit(tmp_path, monkeypatch):
    worker = _carla_worker_module(monkeypatch)

    carla_process = type(
        "OwnedCarla", (), {"pid": 12345, "_carla_pgid": 12345}
    )()
    cleaned = []

    class TerminatingQueue:
        def get(self, timeout=None):
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM handler did not interrupt the worker")

    class ResultQueue:
        def put(self, _payload):
            pass

    monkeypatch.setattr(
        worker,
        "_start_carla_with_retries",
        lambda _cfg, initial_delay_s, process_started: (
            process_started(carla_process) or carla_process
        ),
    )
    monkeypatch.setattr(
        worker,
        "_terminate_carla",
        lambda proc, port: cleaned.append((proc, port)),
    )
    cfg = worker.WorkerConfig(
        worker_id=0,
        port=6000,
        carla_path="/opt/carla/CarlaUE4.sh",
        output_root=str(tmp_path),
        algo_version="parego_qrf_ipc_v3",
        startup_delay_s=0.0,
    )

    with pytest.raises(SystemExit, match="143"):
        worker.worker_main(TerminatingQueue(), ResultQueue(), cfg)

    assert cleaned == [(carla_process, 6000)]


def test_worker_publishes_carla_ownership_before_startup_wait(monkeypatch):
    worker = _carla_worker_module(monkeypatch)

    class RunningCarla:
        pid = 12345
        stderr = None

        def poll(self):
            return None

    carla_process = RunningCarla()
    owned = []
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_args, **_kwargs: carla_process)
    monkeypatch.setattr(
        worker.time,
        "sleep",
        lambda _seconds: owned == [carla_process]
        or pytest.fail("CARLA ownership was not published before startup wait"),
    )
    monkeypatch.setattr(worker, "_start_carla_stderr_drain", lambda *_args: None)

    returned = worker._start_carla(
        "/opt/carla/CarlaUE4.sh",
        6000,
        process_started=owned.append,
    )

    assert returned is carla_process


def test_worker_persists_exact_carla_process_group(tmp_path, monkeypatch):
    worker = _carla_worker_module(monkeypatch)

    cfg = worker.WorkerConfig(
        worker_id=2,
        port=8000,
        carla_path="/opt/carla/CarlaUE4.sh",
        output_root=str(tmp_path),
        algo_version="parego_qrf_ipc_v3",
    )
    proc = type("OwnedCarla", (), {"pid": 12345, "_carla_pgid": 12345})()
    monkeypatch.setattr(worker.os, "getpid", lambda: 54321)
    monkeypatch.setattr(worker.time, "time", lambda: 1000.5)

    worker._publish_carla_ownership(cfg, proc)

    record = json.loads(
        (tmp_path / "carla_ownership/worker_2.json").read_text()
    )
    assert record == {
        "schema_version": 1,
        "worker_id": 2,
        "worker_pid": 54321,
        "port": 8000,
        "carla_path": "/opt/carla/CarlaUE4.sh",
        "launcher_pid": 12345,
        "carla_pgid": 12345,
        "started_unix": 1000.5,
    }


def test_worker_terminates_saved_carla_group_after_launcher_exit(monkeypatch):
    worker = _carla_worker_module(monkeypatch)

    signals = []

    class ExitedLauncher:
        pid = 12345
        _carla_pgid = 12345

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pgid, signum: signals.append((pgid, signum)),
    )
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)

    worker._terminate_carla(ExitedLauncher(), 6000)

    assert signals == [
        (12345, signal.SIGTERM),
        (12345, signal.SIGKILL),
    ]
