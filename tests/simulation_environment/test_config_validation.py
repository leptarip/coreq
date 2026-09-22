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
"""Pin the gate that stops a malformed config becoming a wrong campaign.

`cfg_parser` is the most-imported module in the simulator, and until now it
appeared in the suite exactly once -- stubbed out as a fake whose `parse`
returns `{}`. Nothing executed the real validators.

Two things are covered here. First, that the configs actually shipped in this
repository still pass, which is the regression guard that matters most: a
tightened rule that rejects `intersection2_seeded.toml` would stop every
campaign. Second, the accept/reject matrix rule by rule, including the timing
compatibility arithmetic and the delegation into the network, fault, vehicle and
sensor validators.

Several tests document defects rather than intent. They are named so, and say in
their body what a fix should change.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from source.simulation_environment.cfg import cfg_parser, sensor_cfg
from source.simulation_environment.cfg.sensor_cfg import SensorCfg
from source.simulation_environment.cfg.vehicle_cfg import VehicleCfg
from source.simulation_environment.obps.logic.static_ghost import FovSectorStaticGhost
from source.simulation_environment.sim_constants import ALLOWED_DELTA_TIMES
from source.simulation_environment.sim_utils import get_sim_step


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIGS = [
    "tests/fixtures/intersection1_seeded.toml",
    "tests/fixtures/intersection2_seeded.toml",
    "configuration/config_intersection1_debug.toml",
    "configuration/config_intersection2_debug.toml",
]
REFERENCE_CONFIG = REPOSITORY_ROOT / "tests/fixtures/intersection2_seeded.toml"


@pytest.fixture
def cfg():
    """A parsed copy of a real shipped config, safe to mutate per test."""
    return cfg_parser.parse(str(REFERENCE_CONFIG))


def _position_stub(_name):
    return (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# The configs this repository actually ships
# --------------------------------------------------------------------------


class TestShippedConfigsRemainValid:

    @pytest.mark.parametrize("relative_path", SHIPPED_CONFIGS)
    def test_every_shipped_config_parses_and_validates(self, relative_path):
        parsed = cfg_parser.parse(str(REPOSITORY_ROOT / relative_path))

        assert cfg_parser.validate_cfg(parsed) == (True, "")

    @pytest.mark.parametrize("relative_path", SHIPPED_CONFIGS)
    def test_every_shipped_config_yields_usable_sensors(self, relative_path):
        parsed = cfg_parser.parse(str(REPOSITORY_ROOT / relative_path))

        sensors = cfg_parser.get_sensors_cfg(parsed, _position_stub)

        assert sensors, "a config that builds no sensors runs a blind simulation"
        assert len(sensors) == len(parsed["sensors"])
        assert all(isinstance(entry, SensorCfg) for entry in sensors)


class TestParse:

    def test_it_reads_a_toml_document_into_a_dictionary(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("[sim]\ndelta_time = 0.01\nseeds = [1, 2]\n")

        assert cfg_parser.parse(str(path)) == {"sim": {"delta_time": 0.01, "seeds": [1, 2]}}

    def test_a_missing_file_is_not_silently_swallowed(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cfg_parser.parse(str(tmp_path / "absent.toml"))

    def test_malformed_toml_is_not_silently_swallowed(self, tmp_path):
        path = tmp_path / "bad.toml"
        path.write_text("this is not = = toml")

        with pytest.raises(Exception, match="(?i)expected|invalid|cannot"):
            cfg_parser.parse(str(path))


# --------------------------------------------------------------------------
# Timing compatibility
# --------------------------------------------------------------------------


class TestSimulationTimestep:

    @pytest.mark.parametrize("delta_time", ALLOWED_DELTA_TIMES)
    def test_each_allowed_timestep_is_accepted(self, cfg, delta_time):
        cfg["sim"]["delta_time"] = delta_time
        step = get_sim_step(delta_time)
        cfg["ego"]["decision_period"] = step * 2
        cfg["adversary"]["decision_period"] = step * 2
        for sensor in cfg["sensors"]:
            sensor["gen_time"] = step * 4

        assert cfg_parser.validate_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("delta_time", [0.02, 0.1, 0.0005, 1.0, 0.0])
    def test_a_timestep_outside_the_allowed_set_is_rejected(self, cfg, delta_time):
        cfg["sim"]["delta_time"] = delta_time

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "not in allowed delta times" in reason

    def test_a_non_numeric_timestep_is_caught_by_the_allowed_set_check(self, cfg):
        cfg["sim"]["delta_time"] = "fast"

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "not in allowed delta times" in reason


class TestDecisionPeriodAlignment:
    """A decision period that is not a whole number of simulator steps would
    silently drift against the tick clock."""

    @pytest.mark.parametrize("role", ["ego", "adversary"])
    @pytest.mark.parametrize("period", [10, 20, 100])
    def test_an_aligned_period_is_accepted(self, cfg, role, period):
        cfg[role]["decision_period"] = period

        assert cfg_parser.validate_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("role", ["ego", "adversary"])
    @pytest.mark.parametrize("period", [7, 15, 3])
    def test_a_misaligned_period_is_rejected_and_names_the_role(self, cfg, role, period):
        cfg[role]["decision_period"] = period

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert reason.startswith(role)
        assert "not compatible with sim delta_time" in reason

    def test_alignment_is_measured_against_the_derived_simulator_step(self, cfg):
        """`delta_time` is in seconds and the periods are in milliseconds, so the
        modulus is taken against `delta_time / 0.001`."""
        cfg["sim"]["delta_time"] = 0.005
        assert get_sim_step(0.005) == 5

        cfg["ego"]["decision_period"] = 10
        cfg["adversary"]["decision_period"] = 15
        assert cfg_parser.validate_cfg(cfg)[0]

        cfg["ego"]["decision_period"] = 12
        assert not cfg_parser.validate_cfg(cfg)[0]


class TestSensorGenerationTime:

    @pytest.mark.parametrize("gen_time", [10, 20, 50])
    def test_an_aligned_generation_time_is_accepted(self, cfg, gen_time):
        cfg["sensors"][0]["gen_time"] = gen_time

        assert cfg_parser.validate_cfg(cfg) == (True, "")

    def test_a_misaligned_generation_time_is_rejected_and_names_the_sensor(self, cfg):
        cfg["sensors"][0]["gen_time"] = 15

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "sensor A gen time not compatible" in reason

    def test_every_sensor_is_checked_not_just_the_first(self, cfg):
        second = copy.deepcopy(cfg["sensors"][0])
        second["name"] = "B"
        second["gen_time"] = 15
        cfg["sensors"].append(second)

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "sensor B" in reason


class TestSeedList:

    def test_an_absent_seed_list_is_allowed(self, cfg):
        cfg["sim"].pop("seed_list", None)

        assert cfg_parser.validate_cfg(cfg) == (True, "")

    def test_a_seed_list_matching_the_episode_count_is_accepted(self, cfg):
        cfg["sim"]["num_episodes"] = 3
        cfg["sim"]["seed_list"] = [11, 22, 33]

        assert cfg_parser.validate_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("seeds", [[1], [1, 2, 3, 4], []])
    def test_a_length_mismatch_is_rejected(self, cfg, seeds):
        cfg["sim"]["num_episodes"] = 3
        cfg["sim"]["seed_list"] = seeds

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "seed_list length must match" in reason

    @pytest.mark.parametrize("seeds", ["1,2,3", 7, {"a": 1}])
    def test_a_non_list_seed_list_is_rejected(self, cfg, seeds):
        cfg["sim"]["seed_list"] = seeds

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "seed_list must be a list" in reason


# --------------------------------------------------------------------------
# Delegation
# --------------------------------------------------------------------------


class TestDelegationToSubValidators:
    """`validate_cfg` is a composition. Each sub-validator must be reached, and
    its diagnostic must reach the caller unchanged -- an aggregated "invalid
    config" message would make a bad campaign much harder to diagnose."""

    def test_the_network_validator_is_reached(self, cfg):
        cfg["communication"]["network"]["type"] = "smoke-signal"

        assert cfg_parser.validate_cfg(cfg) == (False, "Unknown network type")

    def test_a_network_model_diagnostic_is_surfaced_verbatim(self, cfg):
        cfg["communication"]["network"]["delay_min"] = -5

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "NetworkLogNorm" in reason

    def test_the_fault_validator_is_reached(self, cfg):
        cfg["communication"]["fault"]["type"] = "gremlins"

        assert cfg_parser.validate_cfg(cfg) == (False, "Unknown fault type")

    def test_a_fault_model_diagnostic_is_surfaced_verbatim(self, cfg):
        cfg["communication"]["fault"].pop("offline")

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "offline" in reason

    @pytest.mark.parametrize("role", ["ego", "adversary"])
    def test_both_vehicle_blocks_are_validated(self, cfg, role):
        cfg[role].pop("target_v")

        assert cfg_parser.validate_cfg(cfg) == (False, "Missing required parameters")

    def test_the_rare_event_validator_is_reached(self, cfg):
        cfg["rare_event"] = {"enabled": True, "defensive_mixture_probability": 2.0}

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "defensive_mixture_probability" in reason


class TestStructuralErrorsCollapseToOneMessage:
    """Characterisation of a real weakness, not of intent.

    `validate_cfg` wraps its whole body in `except Exception`, so any missing
    section becomes the same opaque string and the `KeyError` naming the actual
    section is only printed, never returned. A fix should surface which key was
    missing; these tests would then need updating.
    """

    @pytest.mark.parametrize(
        "section", ["sim", "ego", "adversary", "communication", "sensors"]
    )
    def test_a_missing_top_level_section_loses_its_diagnostic(self, cfg, section):
        cfg.pop(section)

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert reason == "Exception validating config"
        assert section not in reason, "the missing key is not named in the result"

    def test_a_missing_episode_count_beside_a_seed_list_loses_its_diagnostic(self, cfg):
        cfg["sim"]["seed_list"] = [1, 2]
        cfg["sim"].pop("num_episodes")

        assert cfg_parser.validate_cfg(cfg) == (False, "Exception validating config")


# --------------------------------------------------------------------------
# Vehicle block
# --------------------------------------------------------------------------


class TestVehicleCfgValidation:

    def test_a_complete_vehicle_block_is_accepted(self, cfg):
        assert VehicleCfg.validate_cfg(cfg["ego"]) == (True, "")

    @pytest.mark.parametrize(
        "missing", ["target_v", "initial_v", "perception", "decision_period", "role_name"]
    )
    def test_every_required_field_is_required(self, cfg, missing):
        cfg["ego"].pop(missing)

        valid, reason = VehicleCfg.validate_cfg(cfg["ego"])

        assert not valid
        assert reason == "Missing required parameters"

    def test_the_perception_block_must_carry_a_field_of_view(self, cfg):
        cfg["ego"]["perception"].pop("fov")

        assert VehicleCfg.validate_cfg(cfg["ego"]) == (False, "Missing fov parameter")

    @pytest.mark.parametrize("missing", ["degrees", "ray_step", "range"])
    def test_every_field_of_view_parameter_is_required(self, cfg, missing):
        cfg["ego"]["perception"]["fov"].pop(missing)

        assert VehicleCfg.validate_cfg(cfg["ego"]) == (False, "Missing fov parameters")

    def test_an_absent_prediction_window_is_filled_in_with_the_default(self, cfg):
        """The validator mutates the config it is handed. Callers downstream rely
        on the key existing afterwards, so the side effect is load-bearing."""
        cfg["ego"]["perception"].pop("prediction_window")

        valid, _ = VehicleCfg.validate_cfg(cfg["ego"])

        assert valid
        assert cfg["ego"]["perception"]["prediction_window"] == VehicleCfg.DEFAULT_PREDICTION_WINDOW

    def test_an_explicit_prediction_window_is_left_alone(self, cfg):
        cfg["ego"]["perception"]["prediction_window"] = 3.5

        VehicleCfg.validate_cfg(cfg["ego"])

        assert cfg["ego"]["perception"]["prediction_window"] == 3.5


# --------------------------------------------------------------------------
# Rare-event section
# --------------------------------------------------------------------------


class TestRareEventSection:

    def test_an_absent_section_is_treated_as_disabled(self, cfg):
        cfg.pop("rare_event", None)

        assert cfg_parser.validate_rare_event_cfg(cfg) == (True, "")

    def test_an_empty_section_is_treated_as_disabled(self, cfg):
        cfg["rare_event"] = {}

        assert cfg_parser.validate_rare_event_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("flag", ["yes", 1, 0, None, "true"])
    def test_the_enabled_flag_must_be_a_boolean(self, cfg, flag):
        cfg["rare_event"] = {"enabled": flag}

        valid, reason = cfg_parser.validate_rare_event_cfg(cfg)

        assert not valid
        assert reason == "rare_event.enabled must be a boolean"

    def test_a_disabled_section_skips_every_bias_check(self, cfg):
        """Characterisation: nothing under a disabled section is validated, so a
        config can carry bias values that would be rejected once enabled."""
        cfg["rare_event"] = {
            "enabled": False,
            "network_bias": {"delay": -999},
            "fault_bias": {"trigger": 5},
            "sensor_bias": {"miss_scale": -1.0},
        }

        assert cfg_parser.validate_rare_event_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    def test_a_mixture_probability_inside_the_unit_interval_is_accepted(
        self, cfg, probability
    ):
        cfg["rare_event"] = {
            "enabled": True, "defensive_mixture_probability": probability
        }

        assert cfg_parser.validate_rare_event_cfg(cfg) == (True, "")

    @pytest.mark.parametrize("probability", [-0.1, 1.5, 2.0])
    def test_a_mixture_probability_outside_it_is_rejected(self, cfg, probability):
        cfg["rare_event"] = {
            "enabled": True, "defensive_mixture_probability": probability
        }

        valid, reason = cfg_parser.validate_rare_event_cfg(cfg)

        assert not valid
        assert reason == "rare_event.defensive_mixture_probability must be in [0, 1]"

    def test_an_enabled_section_with_no_bias_blocks_is_accepted(self, cfg):
        cfg["rare_event"] = {"enabled": True}

        assert cfg_parser.validate_rare_event_cfg(cfg) == (True, "")

    @pytest.mark.parametrize(
        "block,fragment",
        [
            ({"network_bias": {"delay_min": -1}}, "network_bias.delay_min"),
            ({"fault_bias": {"trigger": 5}}, "Unsupported rare_event.fault_bias"),
            ({"fault_bias": {"trigger_var_scale": 0}}, "trigger_var_scale must be > 0"),
            ({"sensor_bias": {"miss_scale": 0}}, "sensor_bias.miss_scale must be > 0"),
        ],
    )
    def test_each_bias_block_is_delegated_to_its_own_validator(
        self, cfg, block, fragment
    ):
        cfg["rare_event"] = {"enabled": True, **block}

        valid, reason = cfg_parser.validate_rare_event_cfg(cfg)

        assert not valid
        assert fragment in reason

    def test_the_sensor_validator_sees_the_configured_sensors(self, cfg):
        """The shipped sensor uses deterministic `worst_case` uncertainty, which
        cannot carry an importance-sampled offset."""
        cfg["rare_event"] = {"enabled": True, "sensor_bias": {"unc_p_add": 0.3}}

        valid, reason = cfg_parser.validate_rare_event_cfg(cfg)

        assert not valid
        assert "requires stochastic normal uncertainty" in reason
        assert "sensor A" in reason


class TestSensorBiasValidation:

    @staticmethod
    def _normal_sensors(cfg, pos=0.5, vel=1.5):
        sensors = copy.deepcopy(cfg["sensors"])
        sensors[0]["fov"]["sector"]["uncertainty"] = {
            "type": "normal", "pos": pos, "vel": vel
        }
        return sensors

    def test_a_disabled_section_short_circuits(self, cfg):
        block = {"enabled": False, "sensor_bias": {"miss_scale": -1}}

        assert sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"]) == (True, "")

    def test_an_enabled_section_without_sensor_bias_is_accepted(self, cfg):
        assert sensor_cfg.validate_rare_event_cfg({"enabled": True}, cfg["sensors"]) == (
            True, ""
        )

    @pytest.mark.parametrize("scale", [0.0, -1.0, -0.5])
    def test_a_non_positive_scale_is_rejected(self, cfg, scale):
        block = {"enabled": True, "sensor_bias": {"miss_scale": scale}}

        valid, reason = sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"])

        assert not valid
        assert "sensor_bias.miss_scale must be > 0" in reason

    @pytest.mark.parametrize("scale", [0.5, 1.0, 4.0])
    def test_a_positive_scale_is_accepted_even_for_deterministic_uncertainty(
        self, cfg, scale
    ):
        """Detection-probability scaling is independent of the uncertainty model,
        so `worst_case` sensors may still be biased this way."""
        block = {"enabled": True, "sensor_bias": {"miss_scale": scale}}

        assert sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"]) == (True, "")

    def test_per_sensor_audit_scales_are_accepted(self, cfg):
        block = {"enabled": True, "sensor_bias": {"audit_miss_scales": {"A": 2.0}}}

        assert sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"]) == (True, "")

    @pytest.mark.parametrize(
        "audit,fragment",
        [
            ({"": 2.0}, "keys must be non-empty"),
            ({"  ": 2.0}, "keys must be non-empty"),
            ({"A": 0.0}, "values must be > 0"),
            ({"A": -1.0}, "values must be > 0"),
            ([1, 2], "must be a mapping"),
            ("A", "must be a mapping"),
        ],
    )
    def test_malformed_audit_scales_are_rejected(self, cfg, audit, fragment):
        block = {"enabled": True, "sensor_bias": {"audit_miss_scales": audit}}

        valid, reason = sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"])

        assert not valid
        assert fragment in reason

    @pytest.mark.parametrize("field", ["unc_p_add", "unc_v_add"])
    def test_an_uncertainty_offset_needs_a_stochastic_sensor(self, cfg, field):
        block = {"enabled": True, "sensor_bias": {field: 0.3}}

        valid, reason = sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"])

        assert not valid
        assert "requires stochastic normal uncertainty" in reason

    @pytest.mark.parametrize("field", ["unc_p_add", "unc_v_add"])
    def test_an_uncertainty_offset_is_accepted_on_a_normal_sensor(self, cfg, field):
        block = {"enabled": True, "sensor_bias": {field: 0.3}}

        assert sensor_cfg.validate_rare_event_cfg(
            block, self._normal_sensors(cfg)
        ) == (True, "")

    @pytest.mark.parametrize(
        "field,axis", [("unc_p_add", "position"), ("unc_v_add", "velocity")]
    )
    def test_an_offset_that_drives_the_proposal_sigma_non_positive_is_rejected(
        self, cfg, field, axis
    ):
        block = {"enabled": True, "sensor_bias": {field: -9.0}}

        valid, reason = sensor_cfg.validate_rare_event_cfg(
            block, self._normal_sensors(cfg)
        )

        assert not valid
        assert f"positive nominal and proposal {axis} sigmas" in reason

    @pytest.mark.parametrize("field", ["unc_p_add", "unc_v_add"])
    def test_an_offset_on_a_zero_nominal_sigma_is_rejected(self, cfg, field):
        block = {"enabled": True, "sensor_bias": {field: 0.3}}

        valid, reason = sensor_cfg.validate_rare_event_cfg(
            block, self._normal_sensors(cfg, pos=0.0, vel=0.0)
        )

        assert not valid
        assert "positive nominal and proposal" in reason

    def test_with_no_sensors_there_is_nothing_to_contradict(self, cfg):
        block = {"enabled": True, "sensor_bias": {"unc_p_add": 0.3}}

        assert sensor_cfg.validate_rare_event_cfg(block, []) == (True, "")
        assert sensor_cfg.validate_rare_event_cfg(block, None) == (True, "")

    @pytest.mark.parametrize(
        "window",
        [{"start_ms": 5}, {"end_ms": 5}, {"start_ms": 9, "end_ms": 2}, {"start_ms": -1, "end_ms": 5}],
    )
    def test_a_malformed_active_window_is_rejected(self, cfg, window):
        block = {
            "enabled": True,
            "sensor_bias": {"miss_scale": 2.0, "active_window": window},
        }

        valid, reason = sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"])

        assert not valid
        assert "active_window" in reason

    def test_a_well_formed_active_window_is_accepted(self, cfg):
        block = {
            "enabled": True,
            "sensor_bias": {
                "miss_scale": 2.0,
                "active_window": {"start_ms": 0, "end_ms": 900, "until_successes": 3},
            },
        }

        assert sensor_cfg.validate_rare_event_cfg(block, cfg["sensors"]) == (True, "")


# --------------------------------------------------------------------------
# Sensor entries
# --------------------------------------------------------------------------


class TestSensorEntryValidation:
    """Every key `get_sensors_cfg` reads without a default has to be checked
    here, because a missing one is swallowed at construction time and takes the
    whole sensor list with it."""

    COMMON_FIELDS = ["recipient", "gen_time", "stream", "orientation"]
    SECTOR_FIELDS = ["degrees", "probA", "probB", "uncertainty"]

    def test_a_complete_sensor_entry_is_accepted(self, cfg):
        assert SensorCfg.validate_cfg(cfg["sensors"][0]) == (True, "")

    @pytest.mark.parametrize("field", COMMON_FIELDS)
    def test_every_common_field_is_required(self, cfg, field):
        cfg["sensors"][0].pop(field)

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert field in reason

    @pytest.mark.parametrize("field", ["position", "retransmission"])
    def test_sector_only_fields_are_required_for_a_sector_sensor(self, cfg, field):
        cfg["sensors"][0].pop(field)

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert field in reason

    @pytest.mark.parametrize("field", ["queue_size", "policy"])
    def test_both_retransmission_settings_are_required(self, cfg, field):
        cfg["sensors"][0]["retransmission"].pop(field)

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert "retransmission.{0}".format(field) in reason

    @pytest.mark.parametrize("field", ["position", "retransmission"])
    def test_sector_only_fields_are_not_required_for_a_ghost_sensor(self, cfg, field):
        cfg["sensors"][0]["fov"]["type"] = SensorCfg.TYPE_SECTOR_GHOST
        cfg["sensors"][0].pop(field)

        assert SensorCfg.validate_cfg(cfg["sensors"][0]) == (True, "")

    @pytest.mark.parametrize("removed", ["fov", "type"])
    def test_the_field_of_view_type_is_required(self, cfg, removed):
        sensor = cfg["sensors"][0]
        sensor.pop("fov") if removed == "fov" else sensor["fov"].pop("type")

        valid, reason = SensorCfg.validate_cfg(sensor)

        assert not valid
        assert "fov.type" in reason

    @pytest.mark.parametrize("fov_type", ["lidar", "sector_", "", None, 3])
    def test_an_unrecognised_field_of_view_type_is_rejected(self, cfg, fov_type):
        """A typo here used to disable the sensor silently."""
        cfg["sensors"][0]["fov"]["type"] = fov_type

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert "Unknown fov type" in reason

    @pytest.mark.parametrize("fov_type", [SensorCfg.TYPE_SECTOR, SensorCfg.TYPE_SECTOR_GHOST])
    def test_both_supported_types_are_accepted(self, cfg, fov_type):
        cfg["sensors"][0]["fov"]["type"] = fov_type

        assert SensorCfg.validate_cfg(cfg["sensors"][0]) == (True, "")

    @pytest.mark.parametrize("sector", [None, "sector", 3, []])
    def test_the_sector_block_must_be_a_mapping(self, cfg, sector):
        cfg["sensors"][0]["fov"]["sector"] = sector

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert "fov.sector" in reason

    @pytest.mark.parametrize("field", SECTOR_FIELDS)
    def test_every_sector_field_the_geometry_reads_is_required(self, cfg, field):
        cfg["sensors"][0]["fov"]["sector"].pop(field)

        valid, reason = SensorCfg.validate_cfg(cfg["sensors"][0])

        assert not valid
        assert "fov.sector.{0}".format(field) in reason

    def test_the_name_stays_optional(self, cfg):
        """`get_sensors_cfg` defaults it, so it is not a structural requirement."""
        cfg["sensors"][0].pop("name")

        assert SensorCfg.validate_cfg(cfg["sensors"][0]) == (True, "")

    def test_the_diagnostic_names_the_offending_sensor(self, cfg):
        cfg["sensors"][0]["name"] = "front-left"
        cfg["sensors"][0].pop("recipient")

        assert "front-left" in SensorCfg.validate_cfg(cfg["sensors"][0])[1]

    def test_an_unnamed_sensor_still_produces_a_readable_diagnostic(self, cfg):
        cfg["sensors"][0].pop("name")
        cfg["sensors"][0].pop("recipient")

        assert "<unnamed>" in SensorCfg.validate_cfg(cfg["sensors"][0])[1]


class TestMalformedSensorsInvalidateTheConfig:
    """The property the validator exists for: a config that would build no
    sensors must not validate."""

    @pytest.mark.parametrize(
        "field",
        ["recipient", "gen_time", "stream", "orientation", "position", "retransmission", "fov"],
    )
    def test_a_sensor_missing_a_required_field_makes_the_config_invalid(self, cfg, field):
        cfg["sensors"][0].pop(field)

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert reason != "Exception validating config", "the diagnostic must survive"
        assert "sensor" in reason

    def test_a_second_malformed_sensor_makes_the_config_invalid(self, cfg):
        """The case that motivated this: one bad entry silently voided the whole
        sensor list while the config still validated."""
        broken = copy.deepcopy(cfg["sensors"][0])
        broken["name"] = "B"
        broken.pop("recipient")
        cfg["sensors"].append(broken)

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "sensor B" in reason

    def test_an_unrecognised_field_of_view_type_makes_the_config_invalid(self, cfg):
        cfg["sensors"][0]["fov"]["type"] = "lidar"

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "Unknown fov type" in reason

    def test_structure_is_checked_before_timing(self, cfg):
        """A sensor missing `gen_time` used to collapse into the generic
        exception message; it now names the field."""
        cfg["sensors"][0].pop("gen_time")

        valid, reason = cfg_parser.validate_cfg(cfg)

        assert not valid
        assert "gen_time" in reason

    def test_a_config_that_validates_always_builds_every_sensor(self, cfg):
        """The invariant the two halves now share."""
        second = copy.deepcopy(cfg["sensors"][0])
        second["name"] = "B"
        cfg["sensors"].append(second)

        assert cfg_parser.validate_cfg(cfg) == (True, "")
        assert len(cfg_parser.get_sensors_cfg(cfg, _position_stub)) == 2


# --------------------------------------------------------------------------
# Sensor construction
# --------------------------------------------------------------------------


class TestGetSensorsCfg:

    def test_a_sector_sensor_is_built_with_every_field_carried_across(self, cfg):
        built = cfg_parser.get_sensors_cfg(cfg, _position_stub)

        assert len(built) == 1
        sensor = built[0]
        source = cfg["sensors"][0]
        assert sensor.name == source["name"]
        assert sensor.recipient == source["recipient"]
        assert sensor.t_gen == source["gen_time"]
        assert sensor.stream == source["stream"]
        assert sensor.retransmission_queue_size == source["retransmission"]["queue_size"]
        assert sensor.retransmission_queue_policy == source["retransmission"]["policy"]

    def test_the_scenario_callable_resolves_the_named_position(self, cfg):
        seen = []

        def _record(name):
            seen.append(name)
            return (1.0, 2.0, 3.0)

        cfg_parser.get_sensors_cfg(cfg, _record)

        assert seen == [cfg["sensors"][0]["position"]]

    def test_an_unnamed_sensor_falls_back_to_the_empty_name(self, cfg):
        cfg["sensors"][0].pop("name")

        built = cfg_parser.get_sensors_cfg(cfg, _position_stub)

        assert built[0].name == ""

    def test_several_sensors_are_built_in_configuration_order(self, cfg):
        second = copy.deepcopy(cfg["sensors"][0])
        second["name"] = "B"
        cfg["sensors"].append(second)

        built = cfg_parser.get_sensors_cfg(cfg, _position_stub)

        assert [entry.name for entry in built] == ["A", "B"]

    def test_an_unrecognised_field_of_view_type_is_skipped_here(self, cfg):
        """Construction skips what it does not recognise. `validate_cfg` is what
        rejects it -- see `TestSensorEntryValidation` -- so this branch is only
        reached by a caller that skipped validation."""
        second = copy.deepcopy(cfg["sensors"][0])
        second["name"] = "B"
        second["fov"]["type"] = "lidar"
        cfg["sensors"].append(second)

        built = cfg_parser.get_sensors_cfg(cfg, _position_stub)

        assert [entry.name for entry in built] == ["A"]

    def test_one_malformed_sensor_discards_every_other_sensor(self, cfg):
        """The `try` wraps the whole loop and returns `[]`, so a typo in the
        second sensor removes the first one too. `validate_cfg` now rejects such
        a config before it gets here, leaving this as the second line of defence
        for a caller that skipped validation.
        """
        good = copy.deepcopy(cfg["sensors"][0])
        broken = copy.deepcopy(cfg["sensors"][0])
        broken["name"] = "B"
        broken.pop("recipient")
        cfg["sensors"] = [good, broken]

        assert cfg_parser.get_sensors_cfg(cfg, _position_stub) == []

    def test_a_failing_position_callable_discards_every_sensor(self, cfg):
        def _explode(_name):
            raise KeyError("unknown position")

        assert cfg_parser.get_sensors_cfg(cfg, _explode) == []


class TestGhostSensorConstruction:

    def test_a_ghost_field_of_view_records_its_own_type(self, cfg):
        """`FovSectorStaticGhost` used to forward its type as `FovSector`'s
        *position* argument, leaving `type_id` on the plain sector default and
        `position` holding the type string. It now passes the type by keyword.
        """
        fov = FovSectorStaticGhost(cfg["sensors"][0])

        assert fov.type_id == SensorCfg.TYPE_SECTOR_GHOST
        assert fov.position is None, "the ghost class still has no position source"


class TestGhostSensorPathIsBroken:
    """A remaining defect on the `sector_ghost` branch, pinned so a fix is
    noticed.

    It is not reachable from a shipped config -- every sensor in this
    repository uses `fov.type = "sector"` -- but the branch is live code that a
    new config could select.
    """

    def test_a_ghost_sensor_discards_the_entire_sensor_list(self, cfg):
        """`get_sensors_cfg` calls `SensorCfg` with five positional arguments on
        this branch, passing `stream` where `retransmission_queue_size` belongs
        and omitting `retransmission_queue_policy` entirely. The resulting
        `TypeError` is swallowed and every sensor is dropped.
        """
        cfg["sensors"][0]["fov"]["type"] = SensorCfg.TYPE_SECTOR_GHOST

        assert cfg_parser.get_sensors_cfg(cfg, _position_stub) == []

    def test_the_ghost_branch_takes_out_healthy_sensors_beside_it(self, cfg):
        ghost = copy.deepcopy(cfg["sensors"][0])
        ghost["name"] = "B"
        ghost["fov"]["type"] = SensorCfg.TYPE_SECTOR_GHOST
        cfg["sensors"].append(ghost)

        assert cfg_parser.get_sensors_cfg(cfg, _position_stub) == []

    def test_the_underlying_constructor_rejects_five_arguments(self, cfg):
        source = cfg["sensors"][0]

        with pytest.raises(TypeError, match="retransmission_queue_policy"):
            SensorCfg(
                "A", object(), source["recipient"], source["gen_time"], source["stream"]
            )

    def test_a_ghost_sensor_still_passes_validation(self, cfg):
        """The arity defect is a construction-time bug, not a config error, so
        `validate_cfg` has nothing to object to -- which is why the branch has
        to stay pinned here."""
        cfg["sensors"][0]["fov"]["type"] = SensorCfg.TYPE_SECTOR_GHOST

        assert cfg_parser.validate_cfg(cfg) == (True, "")
