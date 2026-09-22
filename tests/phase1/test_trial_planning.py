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

"""Reproducibility and input constraints for frozen portfolio trial plans."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from source.design_exploration.audit.trial_planning import build_trial_plan

SCENARIOS = ("intersection1", "intersection2")


def test_trial_plan_is_deterministic_and_fixes_all_seeds_before_execution():
    kwargs = {
        "accepted_idx": np.asarray([3, 7, 11]),
        "scenario_names": SCENARIOS,
        "n": 8,
        "rng_seed": 20260819,
    }

    first = build_trial_plan(**kwargs)
    second = build_trial_plan(**kwargs)

    assert first == second
    assert [trial["trial_index"] for trial in first] == list(range(8))
    assert all(set(trial["seeds"]) == set(SCENARIOS) for trial in first)
    assert len({seed for trial in first for seed in trial["seeds"].values()}) == 16
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        "09d5804d76d1a05a998c458b0c77d46302376d6227277640d9219d01c3cd4fd9"
    )


@pytest.mark.parametrize("overrides", [
    {"accepted_idx": np.array([])},
    {"accepted_idx": np.array([[1, 2]])},
    {"scenario_names": ()},
    {"scenario_names": ("intersection1", "intersection1")},
    {"n": 0},
])
def test_invalid_trial_plan_inputs_are_rejected(overrides):
    kwargs = dict(accepted_idx=np.array([3, 7, 11]), scenario_names=SCENARIOS, n=8, rng_seed=13)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        build_trial_plan(**kwargs)
