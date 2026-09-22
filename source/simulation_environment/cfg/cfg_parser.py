#!/usr/bin/env python
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
from typing import List, Tuple

try:  # Python 3.11+
    import tomllib as _toml_reader
except ImportError:
    import tomli as _toml_reader

from source.simulation_environment.cfg.sensor_cfg import SensorCfg
import source.simulation_environment.cfg.sensor_cfg as sensor_cfg
from source.simulation_environment.obps.logic.sector_logic import FovSector
import source.simulation_environment.logs.printl
import source.simulation_environment.utils as utils
from source.simulation_environment.obps.logic.static_ghost import FovSectorStaticGhost
import source.simulation_environment.sim_constants as sim_constants
import source.simulation_environment.sim_utils as sim_utils
import source.simulation_environment.communication.fault_models.fault_models_factory as fault_models_factory
import source.simulation_environment.communication.network_models.network_models_factory as network_models_factory
from source.simulation_environment.cfg.vehicle_cfg import VehicleCfg


printl = source.simulation_environment.logs.printl.PrintL("CFG PARSER", enabled=True)


def parse(file_path):
    with open(file_path, "rb") as stream:
        c = _toml_reader.load(stream)
    printl.to_print(message="Config: {0}".format(c))
    return c


def validate_cfg(cfg: dict) -> Tuple[bool, str]:
    try:
        if cfg["sim"]["delta_time"] not in sim_constants.ALLOWED_DELTA_TIMES:
            return False, "sim delta_time {0} not in allowed delta times".format(cfg["sim"]["delta_time"])

        if cfg["sim"]["delta_time"] < sim_constants.MIN_DELTA_TIME:
            return False, "sim delta_time {0} too small".format(cfg["sim"]["delta_time"])

        seed_list = cfg["sim"].get("seed_list")
        if seed_list is not None:
            if not isinstance(seed_list, list):
                return False, "sim seed_list must be a list"
            if len(seed_list) != int(cfg["sim"]["num_episodes"]):
                return False, "sim seed_list length must match sim num_episodes"

        sim_step = sim_utils.get_sim_step(cfg["sim"]["delta_time"])

        if round(cfg["ego"]["decision_period"] % sim_step) != 0:
            return False, "ego decision_period={0} not compatible with sim delta_time={1}".format(
                cfg["ego"]["decision_period"], cfg["sim"]["delta_time"])

        if round(cfg["adversary"]["decision_period"] % sim_step) != 0:
            return False, "adversary decision_period={0} not compatible with sim delta_time={1}".format(
                cfg["adversary"]["decision_period"], cfg["sim"]["delta_time"])

        for sensor in cfg["sensors"]:
            valid, description = SensorCfg.validate_cfg(sensor)
            if not valid:
                return False, description

            if round(sensor["gen_time"] % sim_step) != 0:
                return False, "sensor {0} gen time not compatible with sim delta_time {1}".format(
                    sensor["name"], cfg["sim"]["delta_time"])

        valid, description = network_models_factory.validate_cfg(cfg["communication"])
        if not valid:
            return False, description

        valid, description = fault_models_factory.validate_cfg(cfg["communication"])
        if not valid:
            return False, description

        valid, description = VehicleCfg.validate_cfg(cfg["ego"])
        if not valid:
            return False, description

        valid, description = VehicleCfg.validate_cfg(cfg["adversary"])
        if not valid:
            return False, description

        valid, description = validate_rare_event_cfg(cfg)
        if not valid:
            return False, description

    except Exception as e:
        printl.to_print(message="Exception validating config: {0}".format(e), where=printl.STD_OUT_AND_MEM)
        printl.to_print(message="Exception validating config dict: {0}".format(cfg), where=printl.STD_OUT_AND_MEM)
        return False, "Exception validating config"

    return True, ""


def get_sensors_cfg(cfg: dict, scenario_obps_position_callable) -> List[SensorCfg]:
    cfgs = []
    try:
        for sensor in cfg["sensors"]:
            if sensor["fov"]["type"] == SensorCfg.TYPE_SECTOR:
                position = scenario_obps_position_callable(sensor["position"])
                retransmission_queue_size = sensor["retransmission"]["queue_size"]
                retransmission_queue_policy = sensor["retransmission"]["policy"]
                cfgs.append(
                    SensorCfg(utils.key_or_default(sensor, "name", ""),
                              FovSector(sensor, position),
                              sensor["recipient"],
                              sensor["gen_time"],
                              retransmission_queue_size,
                              retransmission_queue_policy,
                              sensor["stream"]
                              ))
            elif sensor["fov"]["type"] == SensorCfg.TYPE_SECTOR_GHOST:
                cfgs.append(
                    SensorCfg(utils.key_or_default(sensor, "name", ""),
                              FovSectorStaticGhost(sensor),
                              sensor["recipient"],
                              sensor["gen_time"],
                              sensor["stream"]
                              ))
    except Exception as e:
        printl.to_print(message="error parsing sensor config: {0}".format(e))
        return []

    return cfgs


def validate_rare_event_cfg(cfg: dict) -> Tuple[bool, str]:
    """
    Optional validation for rare-event settings. If the section is absent, we
    treat it as disabled and return success.
    """
    re_cfg = cfg.get("rare_event")
    if re_cfg is None:
        return True, ""

    # enabled flag (optional)
    if "enabled" in re_cfg and not isinstance(re_cfg["enabled"], bool):
        return False, "rare_event.enabled must be a boolean"
    if not re_cfg.get("enabled", False):
        return True, ""

    mixture_probability = float(re_cfg.get("defensive_mixture_probability", 0.0))
    if not 0.0 <= mixture_probability <= 1.0:
        return False, "rare_event.defensive_mixture_probability must be in [0, 1]"

    valid, desc = network_models_factory.validate_rare_event_cfg(re_cfg, cfg["communication"])
    if not valid:
        return False, desc

    valid, desc = fault_models_factory.validate_rare_event_cfg(re_cfg, cfg["communication"])
    if not valid:
        return False, desc

    if hasattr(sensor_cfg, "validate_rare_event_cfg"):
        valid, desc = sensor_cfg.validate_rare_event_cfg(re_cfg, cfg.get("sensors", []))
        if not valid:
            return False, desc

    return True, ""
