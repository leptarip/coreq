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
"""Builder CLI, immutable split reuse, and trainer error propagation."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.design_exploration.tools import build_offline_surrogates_from_cache as builder


@pytest.mark.parametrize("argv", [
    ["--unknown-option"], ["--calib-frac", "-0.1"], ["--test-frac", "nan"],
    ["--calib-frac", "0.6", "--test-frac", "0.4"], ["--max-train-workers", "0"],
    ["--sprt-P", "0"], ["--sprt-C", "1"], ["--sprt-p1-factor", "2"],
    ["--sprt-beta-err", "-1"], ["--sprt-max-iters", "0"],
    ["--sprt-batch-size", "0"], ["--d-safe", "inf"],
])
def test_invalid_cli_fails_before_training(argv):
    with pytest.raises(SystemExit) as error:
        builder._parse_args(argv)
    assert error.value.code == 2


def test_cli_calls_do_not_leak_configuration():
    assert builder._parse_args(["--seed", "99", "--skip-qrf"]).seed == 99
    default = builder._parse_args([])
    assert default.seed == 13 and not default.skip_qrf


def test_split_is_reproducible_stratified_and_disjoint():
    labels = [(i, int(builder.TestResult.SAFE if i < 80 else builder.TestResult.UNSAFE))
              for i in range(100)]
    splits = builder._split_labeled(labels, 0.15, 0.2, 13)
    assert splits == builder._split_labeled(labels, 0.15, 0.2, 13)
    assert [len(part) for part in splits] == [65, 15, 20]
    assert sorted(gi for part in splits for gi in part) == list(range(100))
    assert [sum(gi >= 80 for gi in part) for part in splits] == [13, 3, 4]


@pytest.fixture
def cached_builder(tmp_path, monkeypatch):
    spec = builder.design_space.SPACE_SPEC
    labels = [(i, int(builder.TestResult.SAFE if i % 2 else builder.TestResult.UNSAFE))
              for i in range(20)]
    monkeypatch.setattr(builder, "EpisodeManager", lambda **kwargs: object())
    monkeypatch.setattr(builder, "_label_train_indices", lambda *args: labels)
    argv = ["--episode-root", str(tmp_path / "episodes"), "--surrogate-root", str(tmp_path / "models"),
            "--skip-qrf", "--max-train-workers", "1"]
    calls = []
    def train(name, split_path, args, executable):
        calls.append((name, split_path, executable))
        return {"model": str(tmp_path / (name + ".joblib"))}
    monkeypatch.setattr(builder, "_run_trainer_subprocess", train)
    return argv, calls, spec


def test_fresh_build_and_reuse_preserve_labels_and_split(tmp_path, monkeypatch, cached_builder):
    argv, calls, _ = cached_builder
    result = builder.build_offline_surrogates(argv)
    assert [call[0] for call in calls] == ["rf", "nn", "hetgp", "hetgpt"]
    paths = [Path(result[key]) for key in ("split_path", "labels_path")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    def forbidden(*args, **kwargs):
        pytest.fail("Reusing a dataset must not read or relabel cached episodes")
    monkeypatch.setattr(builder, "EpisodeManager", forbidden)
    monkeypatch.setattr(builder, "_label_train_indices", forbidden)
    result2 = builder.build_offline_surrogates(argv + ["--split-path", str(paths[0])])
    assert result2 == result
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
    assert len(calls) == 8


def test_default_builds_create_separate_datasets(cached_builder):
    argv, _, _ = cached_builder
    first = builder.build_offline_surrogates(argv)
    before = Path(first["split_path"]).read_bytes()
    second = builder.build_offline_surrogates(argv)
    assert first["dataset_run_dir"] != second["dataset_run_dir"]
    assert Path(first["split_path"]).read_bytes() == before


@pytest.mark.parametrize("overrides", [["--d-safe", "0.6"], ["--sprt-P", "0.1"],
                                        ["--scenario-name", "intersection1"]])
def test_reuse_rejects_mismatched_policy_without_overwriting(cached_builder, overrides):
    argv, calls, _ = cached_builder
    result = builder.build_offline_surrogates(argv)
    paths = [Path(result[key]) for key in ("split_path", "labels_path")]
    before = [p.read_bytes() for p in paths]
    calls.clear()
    with pytest.raises(ValueError, match="does not match|do not match"):
        builder.build_offline_surrogates(argv + ["--split-path", str(paths[0])] + overrides)
    assert not calls
    assert [p.read_bytes() for p in paths] == before


def test_missing_explicit_split_fails_without_creating_dataset(tmp_path, cached_builder):
    argv, calls, _ = cached_builder
    with pytest.raises(FileNotFoundError):
        builder.build_offline_surrogates(argv + ["--split-path", str(tmp_path / "missing.json")])
    assert not calls and not (tmp_path / "models").exists()


def test_reuse_rejects_overlapping_split(cached_builder):
    argv, calls, _ = cached_builder
    result = builder.build_offline_surrogates(argv)
    path = Path(result["split_path"])
    split = json.loads(path.read_text())
    split["calib"].append(split["train"][0])
    path.write_text(json.dumps(split))
    calls.clear()
    with pytest.raises(ValueError, match="disjoint"):
        builder.build_offline_surrogates(argv + ["--split-path", str(path)])
    assert not calls


def test_failed_training_does_not_publish_success_manifest(monkeypatch, cached_builder):
    argv, _, _ = cached_builder
    def fail(*args):
        raise RuntimeError("synthetic trainer failure")
    monkeypatch.setattr(builder, "_run_trainer_subprocess", fail)
    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        builder.build_offline_surrogates(argv)
    root = Path(builder._parse_args(argv).surrogate_root)
    assert list(root.glob("offline_builder/*/offline_splits.json"))
    assert not list(root.glob("offline_builder/*/offline_artifacts.json"))


def test_qrf_uses_explicit_interpreter_even_when_locally_available(tmp_path, monkeypatch):
    python = tmp_path / "python3.11"
    python.touch()
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: object())
    jobs = builder._trainer_jobs(builder._parse_args(["--python311", str(python)]))
    assert jobs == [(name, sys.executable) for name in ("rf", "nn", "hetgp", "hetgpt")] + [("qrf", str(python))]


def test_qrf_falls_back_to_configured_interpreter(tmp_path, monkeypatch):
    python = tmp_path / "python3.11"
    python.touch()
    monkeypatch.setenv("PYTHON311", str(python))
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: None)
    assert builder._trainer_jobs(builder._parse_args([]))[-1] == ("qrf", str(python))


def test_qrf_uses_current_interpreter_when_available(monkeypatch):
    monkeypatch.delenv("PYTHON311", raising=False)
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: object())
    assert builder._trainer_jobs(builder._parse_args([]))[-1] == ("qrf", sys.executable)


def test_skip_qrf_does_not_require_another_interpreter(monkeypatch):
    monkeypatch.setenv("PYTHON311", "/missing/python")
    assert len(builder._trainer_jobs(builder._parse_args(["--skip-qrf"]))) == 4


def test_trainer_failure_includes_name_and_stderr(monkeypatch, tmp_path):
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(7, cmd, stderr="training diagnostic")
    monkeypatch.setattr(builder.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="rf trainer failed.*exit 7.*\\ntraining diagnostic"):
        builder._run_trainer_subprocess("rf", tmp_path / "split.json", builder._parse_args([]), sys.executable)


@pytest.mark.parametrize("output", ["", "invalid", "[]", "{}"])
def test_trainer_requires_model_artifact(monkeypatch, tmp_path, output):
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(RuntimeError, match="rf trainer"):
        builder._run_trainer_subprocess("rf", tmp_path / "split.json", builder._parse_args([]), sys.executable)
