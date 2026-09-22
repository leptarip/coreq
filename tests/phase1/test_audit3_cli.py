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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reproduction.phase1.run_audit import (
    _load_resume_state,
    _open_checkpoint,
    _validate_checkpoint_raw,
    freeze,
    preflight,
)
from source.design_exploration.commons.design_space import SPACE_SPEC


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, n=1, delta=0.1, P=1.0, alpha_frac=1.0):
    scores = tmp_path / "scores.npy"
    np.save(scores, np.ones(len(SPACE_SPEC.build_grid())), allow_pickle=False)
    scenarios = []
    for name in ("s1", "s2"):
        scenario_path = tmp_path / f"{name}.toml"
        scenario_path.write_text("# offline test fixture\n", encoding="utf-8")
        scenarios.append({"name": name, "config": str(scenario_path)})
    declaration = {
        "method": "bonferroni", "global_delta": delta,
        "cache_reuse": "matching_design_id",
        "stages": [{
            "id": "primary", "tau": 0.5, "delta": delta, "n": n,
            "rng_seed": 123, "run_if": "always",
        }],
    }
    declaration_path = tmp_path / "declaration.json"
    _write_json(declaration_path, declaration)
    config = {
        "schema_version": 1, "declaration_path": str(declaration_path),
        "design_space": "v3", "scenarios": scenarios,
        "surrogate": {"mode": "precomputed_f_min", "path": str(scores)},
        "statistics": {"P": P, "alpha_frac": alpha_frac, "d_safe": 0.5},
        "paths": {"execution_root": str(tmp_path / "runs")}, "runtime": {},
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return config_path, tmp_path / "freeze"


def test_preflight_rejects_a_changed_candidate_set(tmp_path):
    config, root = _fixture(tmp_path)
    freeze(config, root)
    np.save(root / "accepted_primary.npy", np.asarray([0, 1], dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="candidate set differs"):
        preflight(root)


def test_freeze_refuses_stage_that_cannot_pass_at_k_zero(tmp_path):
    config, root = _fixture(
        tmp_path, n=1, delta=0.005, P=0.05, alpha_frac=0.05
    )
    with pytest.raises(ValueError, match="can never pass"):
        freeze(config, root)
    assert not root.exists()


class _RawManager:
    def __init__(self, rows):
        self.rows = rows

    def list_gidx_with_counts(self):
        return sorted(self.rows)

    def episodes(self, gidx, consume=False):
        assert not consume
        return self.rows[gidx]


def test_checkpoint_raw_validation_rejects_wrong_design(tmp_path):
    plan = [{"trial_index": 0, "gidx": 4, "seeds": {"s1": 10}}]
    results = {(0, "s1"): {"ep_uid": "u1"}}
    managers = {"s1": _RawManager({5: [{"ep_uid": "u1"}]})}
    with pytest.raises(ValueError, match="no matching durable raw record"):
        _validate_checkpoint_raw(results, plan, managers)


def test_checkpoint_identity_is_bound_to_stage(tmp_path):
    try:
        import sqlite3  # noqa: F401
    except ImportError:
        pytest.skip("Python was built without sqlite3")
    path = tmp_path / "checkpoint.sqlite3"
    connection, results = _open_checkpoint(path, stage_id="a", tau=0.5, n=10)
    connection.close()
    assert results == {}
    with pytest.raises(ValueError, match="identity mismatch"):
        _open_checkpoint(path, stage_id="a", tau=0.6, n=10)


def test_resume_state_is_bound_to_exact_freeze(tmp_path):
    config, freeze_root = _fixture(tmp_path)
    freeze(config, freeze_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "status": "failed", "error": "interrupted", "stages": {},
        "freeze_root": str(freeze_root.resolve()),
    }
    _write_json(run_dir / "run_state.json", state)
    loaded = _load_resume_state(run_dir, freeze_root)
    assert loaded["status"] == "running"
    assert "error" not in loaded

    state["freeze_root"] = str(tmp_path / "another_freeze")
    _write_json(run_dir / "run_state.json", state)
    with pytest.raises(ValueError, match="different freeze"):
        _load_resume_state(run_dir, freeze_root)
