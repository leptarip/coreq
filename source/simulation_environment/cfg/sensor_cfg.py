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

import random
from typing import Tuple

from source.simulation_environment.configuration_error import SimulationConfigurationError


def sample_uncertainty(
    nominal, proposal_add, uncertainty_type, sampler=None,
    *, component="sensor.uncertainty",
):
    """Return a signed bias or draw a zero-mean stochastic sensor error.

    ``worst_case`` is a deterministic, signed calibration bias. ``normal``
    interprets the nominal value as a standard deviation, which must be
    non-negative. Proposal additions belong exclusively to importance-sampled
    normal uncertainty and therefore require a sampler and strictly positive
    base/proposal standard deviations.
    """
    nominal = float(nominal)
    proposal_add = float(proposal_add)
    if uncertainty_type == "worst_case":
        if proposal_add != 0.0:
            raise SimulationConfigurationError(
                "Importance sampling cannot bias deterministic worst_case uncertainty."
            )
        return nominal
    if uncertainty_type != "normal":
        raise SimulationConfigurationError(
            f"Unknown sensor uncertainty type {uncertainty_type!r}."
        )
    if nominal < 0.0:
        raise SimulationConfigurationError(
            "Normal sensor uncertainty requires a non-negative nominal standard deviation."
        )
    if proposal_add != 0.0 and sampler is None:
        raise SimulationConfigurationError(
            "Normal uncertainty proposal_add requires an importance sampler so its "
            "likelihood ratio is recorded."
        )
    proposal = nominal + proposal_add
    if proposal < 0.0:
        raise SimulationConfigurationError(
            "Normal sensor uncertainty requires a non-negative proposal standard deviation."
        )
    if proposal == 0.0:
        return 0.0
    if sampler:
        if nominal <= 0.0 or proposal <= 0.0:
            raise SimulationConfigurationError(
                "Importance-sampled normal uncertainty requires positive nominal and "
                "proposal standard deviations."
            )
        return sampler.normal(
            mu_base=0.0, sigma_base=nominal,
            mu_bias=0.0, sigma_bias=proposal,
            component=component,
        )
    return random.normalvariate(0.0, proposal)


class Fov:
    def __init__(self, type_id, position, orientation):
        self.type_id = type_id
        self.orientation = orientation
        self.position = position


class SensorCfg:
    TYPE_SECTOR = "sector"
    TYPE_SECTOR_GHOST = "sector_ghost"
    ALL_FOV_TYPES = (TYPE_SECTOR, TYPE_SECTOR_GHOST)

    # default probabilities
    DEFAULT_P_DETECTION = 0.7
    DEFAULT_P_GHOST = 0.3
    DEFAULT_P_MISS = 0.2
    DEFAULT_T_GEN = 10

    def __init__(self,
                 name: str,
                 fov: Fov,
                 recipient: str,
                 t_gen,
                 retransmission_queue_size,
                 retransmission_queue_policy,
                 stream=""):
        self.t_gen = t_gen
        self.recipient = recipient
        self.stream = stream
        self.fov = fov
        self.name = name
        self.retransmission_queue_size = retransmission_queue_size
        self.retransmission_queue_policy = retransmission_queue_policy

    @staticmethod
    def validate_cfg(cfg: dict) -> Tuple[bool, str]:
        """Check that a sensor entry carries every field it is built from.

        `get_sensors_cfg` reads these keys without a default and swallows the
        resulting KeyError by returning no sensors at all, so a single malformed
        entry would otherwise leave the simulation running blind. Reject the
        configuration here instead.
        """
        header = "sensor {0} -->".format(cfg.get("name", "<unnamed>"))

        for key in ("recipient", "gen_time", "stream", "orientation"):
            if key not in cfg:
                return False, "{0} Missing '{1}'".format(header, key)

        if "fov" not in cfg or "type" not in cfg["fov"]:
            return False, "{0} Missing 'fov.type'".format(header)

        fov_type = cfg["fov"]["type"]
        if fov_type not in SensorCfg.ALL_FOV_TYPES:
            return False, "{0} Unknown fov type '{1}'".format(header, fov_type)

        sector = cfg["fov"].get("sector")
        if not isinstance(sector, dict):
            return False, "{0} Missing 'fov.sector'".format(header)

        for key in ("degrees", "probA", "probB", "uncertainty"):
            if key not in sector:
                return False, "{0} Missing 'fov.sector.{1}'".format(header, key)

        if fov_type == SensorCfg.TYPE_SECTOR:
            if "position" not in cfg:
                return False, "{0} Missing 'position'".format(header)

            retransmission = cfg.get("retransmission")
            if not isinstance(retransmission, dict):
                return False, "{0} Missing 'retransmission'".format(header)

            for key in ("queue_size", "policy"):
                if key not in retransmission:
                    return False, "{0} Missing 'retransmission.{1}'".format(header, key)

        return True, ""


def validate_rare_event_cfg(re_cfg: dict, sensors=None):
    """
    Validate optional rare_event.sensor_bias section.
    """
    if not re_cfg.get("enabled", False):
        return True, ""

    sensor_bias = re_cfg.get("sensor_bias", {})
    if not sensor_bias:
        return True, ""

    from source.simulation_environment.rare_events.window import validate_bias_window
    valid, reason = validate_bias_window(sensor_bias, "rare_event.sensor_bias")
    if not valid:
        return valid, reason

    ms = sensor_bias.get("miss_scale")
    if ms is not None and ms <= 0:
        return False, "rare_event.sensor_bias.miss_scale must be > 0"
    audit_scales = sensor_bias.get("audit_miss_scales", {}) or {}
    if not isinstance(audit_scales, dict):
        return False, "rare_event.sensor_bias.audit_miss_scales must be a mapping"
    if any(not str(name).strip() for name in audit_scales):
        return False, "rare_event.sensor_bias.audit_miss_scales keys must be non-empty"
    if any(float(scale) <= 0.0 for scale in audit_scales.values()):
        return False, "rare_event.sensor_bias.audit_miss_scales values must be > 0"

    unc_p_add = float(sensor_bias.get("unc_p_add", 0.0))
    unc_v_add = float(sensor_bias.get("unc_v_add", 0.0))
    for sensor in sensors or []:
        sector = sensor.get("fov", {}).get("sector", {})
        uncertainty = sector.get("uncertainty", {})
        uncertainty_type = uncertainty.get("type", "normal")
        if uncertainty_type != "normal" and (unc_p_add != 0.0 or unc_v_add != 0.0):
            return False, (
                "rare_event sensor uncertainty bias requires stochastic normal uncertainty; "
                f"sensor {sensor.get('name', '<unnamed>')} uses {uncertainty_type!r}"
            )
        pos = float(uncertainty.get("pos", 0.0))
        vel = float(uncertainty.get("vel", 0.0))
        if unc_p_add != 0.0 and (pos <= 0.0 or pos + unc_p_add <= 0.0):
            return False, (
                "rare_event.sensor_bias.unc_p_add requires positive nominal and proposal "
                "position sigmas"
            )
        if unc_v_add != 0.0 and (vel <= 0.0 or vel + unc_v_add <= 0.0):
            return False, (
                "rare_event.sensor_bias.unc_v_add requires positive nominal and proposal "
                "velocity sigmas"
            )

    return True, ""
