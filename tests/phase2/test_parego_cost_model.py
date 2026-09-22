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

"""Frozen engineering-cost model and hypervolume references for Phase 2."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from source.design_exploration.commons.design_space import SPACE_SPEC
from source.design_exploration.optimization.cost_model import (
    ENGINEERING_COST_MODEL_ID,
    build_cost_model,
    cost_model_manifest,
    resolve_hypervolume_reference,
)
from source.design_exploration.optimization.parego_qrf_ipc_client import (
    _hypervolume_2d_min_strict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COST_PROFILES = REPO_ROOT / "configuration" / "cost_profiles.json"
ACCEPTED = REPO_ROOT / "results" / "phase1" / "audited_set" / "accepted_v3.npy"


def test_central_profile_cost_range_over_the_design_grid():
    model = build_cost_model(SPACE_SPEC, config_path=str(COST_PROFILES), profile_id="central")
    manifest = cost_model_manifest(model, SPACE_SPEC)

    assert manifest["model_id"] == ENGINEERING_COST_MODEL_ID
    assert manifest["profile_id"] == "central"
    assert (manifest["cost_min"], manifest["cost_max"]) == (100.0, 249.0)


def test_profile_hypervolume_references_dominate_the_audited_set():
    candidate_indices = np.load(ACCEPTED, allow_pickle=False)
    expected = {"compressed": 211.0, "central": 274.0, "bias_premiums_zero": 252.0, "expanded": 338.0}
    for profile_id, cost_reference in expected.items():
        model = build_cost_model(SPACE_SPEC, config_path=str(COST_PROFILES), profile_id=profile_id)
        resolved = resolve_hypervolume_reference(model, SPACE_SPEC, candidate_indices)
        assert resolved["resolved_point"] == [cost_reference, 0.0]
        assert resolved["cost_reference"]["candidate_cost_max"] < cost_reference


def test_hypervolume_rejects_a_reference_that_clips_any_point():
    with np.testing.assert_raises_regex(ValueError, "does not weakly dominate"):
        _hypervolume_2d_min_strict(
            np.asarray([100.0, 306.65]),
            np.asarray([-4.0, -8.0]),
            np.asarray([275.0, 0.0]),
        )

