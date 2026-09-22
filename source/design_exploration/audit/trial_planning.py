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
"""Freeze sampled designs and episode seeds before portfolio audit execution."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from source.design_exploration.commons.fresh_seed_policy import FreshSeedPolicy


def build_trial_plan(
    *,
    accepted_idx: np.ndarray,
    scenario_names: Sequence[str],
    n: int,
    rng_seed: int,
) -> List[Dict[str, object]]:
    """Fix every sampled design and episode seed before any outcome is observed."""
    accepted = np.asarray(accepted_idx, dtype=np.int64)
    if accepted.ndim != 1 or accepted.size == 0:
        raise ValueError("accepted_idx must be a non-empty one-dimensional array")
    if int(n) <= 0:
        raise ValueError("n must be positive")
    if not scenario_names or len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario_names must be non-empty and unique")

    rng = np.random.default_rng(int(rng_seed))
    seed_policy = FreshSeedPolicy(int(rng_seed))
    plan: List[Dict[str, object]] = []
    used_seeds: set[int] = set()
    for trial_index in range(int(n)):
        gidx = int(rng.choice(accepted))
        seeds = {
            scenario_name: int(seed_policy.next_seed(gidx, scenario_name))
            for scenario_name in scenario_names
        }
        collisions = used_seeds.intersection(seeds.values())
        if collisions:
            raise RuntimeError(f"FreshSeedPolicy produced duplicate seeds: {sorted(collisions)}")
        used_seeds.update(seeds.values())
        plan.append({"trial_index": trial_index, "gidx": gidx, "seeds": seeds})
    return plan
