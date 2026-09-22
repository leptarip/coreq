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

"""Tests for engineering cost profiles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

from source.design_exploration.commons.design_space import Design, SPACE_SPEC
from source.design_exploration.optimization.cost_model import (
    engineering_component_costs,
    evaluate_engineering_cost,
    validate_engineering_cost_config,
)

COST_PROFILES = Path(__file__).resolve().parents[2] / "configuration" / "cost_profiles.json"


@pytest.fixture
def config():
    return json.loads(COST_PROFILES.read_text(encoding="utf-8"))



def _design(**updates):
    values = {
        "prob": 0.75,
        "gen_t": 100.0,
        "fault": 2000.0,
        "unc_p": 1.0,
        "unc_v": 2.0,
        "ego_ad_period": 50.0,
        "packet_drop_rate": 0.01,
        "network.delay_min": 50.0,
        "network.delay_avg": 100.0,
        "network.jit": 100.0,
    }
    values.update(updates)
    return Design(values)


def test_central_profile_has_bounded_endpoints(config):
    cheapest = _design()
    most_capable = _design(
        prob=0.975,
        gen_t=20.0,
        fault=0.0,
        unc_p=0.0,
        unc_v=0.0,
        ego_ad_period=10.0,
        packet_drop_rate=0.001,
        **{
            "network.delay_min": 20.0,
            "network.delay_avg": 25.0,
            "network.jit": 10.0,
        }
    )

    assert evaluate_engineering_cost(cheapest, config, "central") == 100.0
    assert evaluate_engineering_cost(most_capable, config, "central") == 249.0


def test_signed_biases_have_symmetric_procurement_cost(config):
    for key, magnitudes in (("unc_p", (0.2, 0.5, 1.0)), ("unc_v", (0.2, 0.5, 1.0, 1.5, 2.0))):
        for magnitude in magnitudes:
            positive = evaluate_engineering_cost(_design(**{key: magnitude}), config)
            negative = evaluate_engineering_cost(_design(**{key: -magnitude}), config)
            assert positive == negative


def test_sensor_bias_semantics_are_frozen_as_residual_sensor_grades(config):
    semantics = config["sensor_bias_semantics"]

    assert semantics["meaning"] == (
        "fixed_signed_residual_systematic_measurement_bias"
    )
    assert semantics["design_identity"] == "signed_unc_p_unc_v_tuple"
    assert semantics["cost_basis"] == "absolute_residual_bias_grade"
    assert semantics["sign_cost_policy"] == "equal_magnitude_equal_premium"
    assert semantics["performance_separation"] == (
        "cost_does_not_encode_behavioral_harm"
    )


def test_zero_bias_premium_profile_retains_designs_but_removes_only_their_cost(config):
    design = _design(unc_p=0.0, unc_v=0.0)
    central = engineering_component_costs(design, config, "central")
    sensitivity = engineering_component_costs(design, config, "bias_premiums_zero")

    assert central["position_residual_bias_grade"] == 10.0
    assert central["velocity_residual_bias_grade"] == 10.0
    assert sensitivity["position_residual_bias_grade"] == 0.0
    assert sensitivity["velocity_residual_bias_grade"] == 0.0
    assert {
        key: value
        for key, value in central.items()
        if key not in (
            "position_residual_bias_grade",
            "velocity_residual_bias_grade",
        )
    } == {
        key: value
        for key, value in sensitivity.items()
        if key not in (
            "position_residual_bias_grade",
            "velocity_residual_bias_grade",
        )
    }
    assert evaluate_engineering_cost(design, config, "central") == (
        evaluate_engineering_cost(design, config, "bias_premiums_zero") + 20.0
    )


def test_recovery_capability_is_finite_and_monotone(config):
    interruption_levels = (2000.0, 1000.0, 500.0, 200.0, 0.0)
    costs = [evaluate_engineering_cost(_design(fault=value), config) for value in interruption_levels]

    assert costs == sorted(costs)
    assert costs[-1] - costs[0] == 50.0
    assert np.isfinite(costs).all()


def test_locked_network_profile_is_priced_once(config):
    premium = engineering_component_costs(
        _design(
            **{
                "network.delay_min": 20.0,
                "network.delay_avg": 25.0,
                "network.jit": 10.0,
            }
        ),
        config,
        "central",
    )

    assert premium["communication_service_tier"] == 25.0
    assert not any(component_id in premium for component_id in ("network_delay", "network_jitter"))


def test_all_profiles_cover_v3_and_respect_the_ratio_bound(config):
    grid = SPACE_SPEC.build_grid()
    maximum_ratio = config["bounds"]["maximum_total_to_base_ratio"]

    for profile_id in config["profiles"]:
        costs = np.asarray([
            evaluate_engineering_cost(Design(zip(SPACE_SPEC.keys, row)), config, profile_id)
            for row in grid
        ])
        assert costs.shape == (27720,)
        assert np.isfinite(costs).all()
        assert costs.min() == config["base_cost"]
        assert costs.max() / config["base_cost"] <= maximum_ratio


def test_nomenclature_keeps_storage_compatibility(config):
    assert config["nomenclature"] == {
        "stored_design_key": "fault",
        "scientific_name": "guaranteed_service_interruption_ms",
        "cost_component": "recovery_capability_premium",
    }


def test_config_rejects_a_missing_subsystem_multiplier(config):
    del config["profiles"]["central"]["subsystem_multipliers"]["recovery"]

    with np.testing.assert_raises(ValueError):
        validate_engineering_cost_config(config, SPACE_SPEC)


def test_config_rejects_a_non_dominating_hypervolume_multiplier(config):
    config["hypervolume_policy"]["cost_reference"]["multiplier"] = 1.0

    with np.testing.assert_raises_regex(ValueError, "must exceed one"):
        validate_engineering_cost_config(config, SPACE_SPEC)


def test_config_rejects_an_unknown_component_multiplier(config):
    config["profiles"]["bias_premiums_zero"]["component_multipliers"] = {
        "unknown_component": 0.0,
    }

    with np.testing.assert_raises_regex(ValueError, "unknown component"):
        validate_engineering_cost_config(config, SPACE_SPEC)
