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

"""The fixed ParEGO policy and the Phase 2 launcher defaults."""
from __future__ import annotations

import inspect

from source.design_exploration.optimization import launch_parego
from source.design_exploration.optimization.parego_qrf_common import (
    PROFILE,
    PROFILE_SPEC,
    ParEGOQuantileRF,
)
from source.design_exploration.optimization.parego_qrf_ipc_client import (
    run_parego_over_candidates,
)


def test_budget_is_ten_initial_plus_twenty_acquired_designs_at_r20():
    assert launch_parego.DEFAULTS["budget"] == 30
    assert launch_parego.DEFAULTS["episodes_per_scenario"] == 20
    assert PROFILE_SPEC.n_initial_points == 10


def test_policy_is_scalarized_lower_quantile_with_seeded_ties():
    assert PROFILE == "scalarized_leaf3_seeded"
    assert PROFILE_SPEC.acquisition == "lower_quantile"
    assert PROFILE_SPEC.surrogate_target == "scalarized_objectives"
    assert PROFILE_SPEC.tie_break == "seeded_random"
    assert launch_parego.POLICY["rf_min_samples_leaf"] == 3
    assert launch_parego.POLICY["rf_quantile"] == 0.25


def test_entry_points_default_to_the_fixed_profile():
    for callable_ in (ParEGOQuantileRF, run_parego_over_candidates):
        assert inspect.signature(callable_).parameters["profile"].default is None


def test_cli_overrides_only_run_settings():
    args = launch_parego._build_arg_parser().parse_args(
        ["--random-seed", "37", "--episode-seed-offset", "600000000", "--budget", "40"]
    )
    assert (args.random_seed, args.episode_seed_offset, args.budget) == (37, 600_000_000, 40)
    assert not hasattr(args, "profile")
    assert not hasattr(args, "rf_min_samples_leaf")
