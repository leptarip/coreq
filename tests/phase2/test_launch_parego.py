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

"""Offline tests for the Phase 2 launcher: audited-set input, run manifests and re-exec."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from source.design_exploration.optimization import launch_parego
from source.design_exploration.optimization.launch_parego import (
    _REPO_ROOT,
    _stop_optimizer_child,
    build_resume_manifest,
    build_run_manifest,
    load_audited_set,
    main,
    run_child_from_manifest,
)
from source.design_exploration.optimization.parego_qrf_common import PROFILE
from source.design_exploration.optimization.parego_qrf_ipc_client import _partitioned_episode_seed
from source.design_exploration.commons.design_space import SPACE_SPEC

BUNDLE = _REPO_ROOT / "results" / "phase1" / "audited_set"
INTERPRETERS = ["--python311", sys.executable, "--python38", sys.executable]


def _config(tmp_path: Path, **overrides):
    args = launch_parego._build_arg_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--prepare-only", *INTERPRETERS]
    )
    cfg = vars(args)
    cfg.update(overrides)
    return cfg


def _copy_bundle(tmp_path: Path) -> Path:
    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copy)
    return copy


def test_shipped_bundle_yields_the_audited_set():
    audit_result, candidate_indices = load_audited_set(BUNDLE)
    assert audit_result["design_space"] == "v3"
    assert candidate_indices.size == 11896
    assert np.array_equal(candidate_indices, np.load(BUNDLE / "accepted_v3.npy"))


@pytest.mark.parametrize("damage", ["f_min", "accepted"])
def test_damaged_audited_set_bundle_is_rejected(tmp_path, damage):
    bundle = _copy_bundle(tmp_path)
    if damage == "f_min":
        scores = np.load(bundle / "f_min_v3.npy")
        scores[np.load(bundle / "accepted_v3.npy")[0]] = 0.0
        np.save(bundle / "f_min_v3.npy", scores)
    else:
        np.save(bundle / "accepted_v3.npy", np.load(bundle / "accepted_v3.npy")[1:])
    with pytest.raises(RuntimeError):
        load_audited_set(bundle)


def test_prepared_run_freezes_policy_seeds_and_references(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("prepare-only must not spawn a child")
    )
    main(["--run-dir", str(tmp_path / "run"), "--prepare-only", *INTERPRETERS])
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    opt = manifest["optimization"]

    assert manifest["candidate_count"] == 11896
    assert opt["profile"] == PROFILE
    assert opt["resolved_policy"]["n_initial_points"] == 10
    assert (opt["rf_min_samples_leaf"], opt["rf_quantile"], opt["rho"]) == (3, 0.25, 0.05)
    assert (opt["budget"], opt["episodes_per_scenario"]) == (30, 20)
    assert (opt["random_seed"], opt["episode_seed_offset"]) == (11, 400_000_000)
    assert opt["cost_model"]["profile_id"] == "central"
    assert opt["hv_ref_point"] == [274.0, 0.0]
    assert np.array_equal(np.load(manifest["candidate_indices_path"]), np.load(BUNDLE / "accepted_v3.npy"))


def test_default_episode_seed_range_is_collision_free():
    final_seed = _partitioned_episode_seed(
        400_000_000, len(SPACE_SPEC.build_grid()) - 1, 1, 19, scenario_count=2, repetitions=20
    )
    assert final_seed == 401_108_799


def test_resume_manifest_reuses_the_frozen_run(tmp_path):
    source_path = build_run_manifest(_config(tmp_path))
    resume_path = build_resume_manifest(
        _config(tmp_path, resume_run_dir=str(source_path.parent), budget=50)
    )
    source = json.loads(source_path.read_text())
    resumed = json.loads(resume_path.read_text())

    assert resumed["candidate_indices_path"] == source["candidate_indices_path"]
    assert resumed["optimization"]["budget"] == 50
    assert resumed["optimization"]["resume"] is True
    assert resumed["optimization"]["random_seed"] == source["optimization"]["random_seed"]


def test_child_passes_the_frozen_run_to_the_client(tmp_path, monkeypatch):
    manifest_path = build_run_manifest(_config(tmp_path, random_seed=23, episode_seed_offset=500_000_000))
    observed = {}
    monkeypatch.setattr(
        "source.design_exploration.optimization.parego_qrf_ipc_client.run_parego_over_candidates",
        lambda **kwargs: observed.update(kwargs),
    )
    run_child_from_manifest(manifest_path)

    assert observed["profile"] == PROFILE
    assert (observed["random_seed"], observed["episode_seed_offset"]) == (23, 500_000_000)
    assert observed["hv_ref_point"] == [274.0, 0.0]
    assert observed["candidate_indices"].size == 11896
    assert [spec.name for spec in observed["scenario_specs"]] == ["intersection1", "intersection2"]


def test_parent_reexecs_client_as_repo_module_with_repo_pythonpath(tmp_path, monkeypatch):
    agents_root = tmp_path / "carla-python-api"
    (agents_root / "agents").mkdir(parents=True)
    observed = {}

    class CompletedChild:
        pid = 12345

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def capture_child(cmd, **kwargs):
        observed["cmd"] = cmd
        observed.update(kwargs)
        return CompletedChild()

    monkeypatch.setattr(launch_parego.subprocess, "Popen", capture_child)
    main(["--run-dir", str(tmp_path / "run"), *INTERPRETERS,
          "--carla-path", sys.executable, "--carla-agents-path", str(agents_root)])

    assert observed["cmd"][1:] == ["-m", "source.design_exploration.optimization.launch_parego"]
    assert Path(observed["cwd"]).resolve() == _REPO_ROOT
    assert observed["start_new_session"] is True
    pythonpath = observed["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(_REPO_ROOT) in pythonpath and str(agents_root.resolve()) in pythonpath
    assert observed["env"][launch_parego.CHILD_ENV] == str(tmp_path / "run" / "run_manifest.json")


def test_launch_requires_a_real_carla_executable(tmp_path):
    with pytest.raises(FileNotFoundError, match="--carla-path"):
        main(["--run-dir", str(tmp_path / "run"), *INTERPRETERS,
              "--carla-path", str(tmp_path / "missing-CarlaUE4.sh")])


def test_optimizer_child_cleanup_escalates_from_interrupt_to_kill(monkeypatch):
    observed_signals = []

    class StubbornChild:
        pid = 43210

        def __init__(self):
            self.wait_count = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.wait_count < 3:
                raise subprocess.TimeoutExpired("optimizer", timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(os, "getpgid", lambda _pid: 43210)
    monkeypatch.setattr(os, "killpg", lambda _pgid, signum: observed_signals.append(signum))
    _stop_optimizer_child(StubbornChild(), interrupt_timeout=0.01, terminate_timeout=0.01)

    assert observed_signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_parent_sigint_is_forwarded_to_owned_optimizer_child(tmp_path):
    child_script = tmp_path / "interruptible_child.py"
    outer_script = tmp_path / "optimizer_parent.py"
    child_pid_path = tmp_path / "child.pid"
    child_stopped_path = tmp_path / "child.stopped"
    child_script.write_text(
        """\
import os
import signal
import sys
import time
from pathlib import Path

def stop(signum, _frame):
    Path(sys.argv[2]).write_text(str(signum), encoding="utf-8")
    raise SystemExit(128 + signum)

signal.signal(signal.SIGINT, stop)
Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    outer_script.write_text(
        """\
import os
import sys
from source.design_exploration.optimization.launch_parego import _run_optimizer_child

_run_optimizer_child([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]], env=dict(os.environ))
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    outer = subprocess.Popen(
        [sys.executable, str(outer_script), str(child_script), str(child_pid_path), str(child_stopped_path)],
        cwd=str(_REPO_ROOT), env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_path.exists(), "optimizer child did not start"
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        os.kill(outer.pid, signal.SIGINT)
        outer.wait(timeout=10.0)
        assert child_stopped_path.read_text(encoding="utf-8") == str(int(signal.SIGINT))
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if outer.poll() is None:
            os.killpg(os.getpgid(outer.pid), signal.SIGKILL)
            outer.wait(timeout=5.0)
