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
import sys
from typing import Tuple, Optional

from source.simulation_environment.communication.fault_models.fault_logic_base import FaultLogicBase, ComEvent
from source.simulation_environment.configuration_error import SimulationConfigurationError
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    require_sampler_for_bias,
)

class SITLogic(FaultLogicBase):
    DEFAULT_OFFLINE = 0
    DEFAULT_TRIGGER = sys.maxsize

    def __init__(self, delta_time: float, cfg: dict,
                 sampler: Optional[ImportanceSampler] = None,
                 bias_cfg: Optional[dict] = None):
        super().__init__()
        require_sampler_for_bias(sampler, bias_cfg, "Fault")
        if bias_cfg and "trigger" in bias_cfg:
            raise SimulationConfigurationError(
                "fault_bias.trigger is a deterministic timing override without a "
                "likelihood-ratio correction; bias trigger_mean or trigger_var instead."
            )
        self.cfg = cfg
        self.time_to_stay_offline = cfg.get("offline", self.DEFAULT_OFFLINE)
        self.trigger_mean = cfg.get("trigger_mean", self.DEFAULT_TRIGGER)
        self.trigger_var = cfg.get("trigger_var", self.DEFAULT_TRIGGER)
        self.trigger = cfg.get("trigger", self.DEFAULT_TRIGGER)
        self.unit_step = delta_time / 0.001  # get simulation step in ms
        self.online = True
        self.start_offline_time = -1
        self.offline_time = -1
        self.triggered = False if self.time_to_stay_offline > 0 else True  # 0 means NO off-line/always on-line
        self.sampler = sampler
        self.bias_cfg = bias_cfg or {}
        self.trigger_step = self._compute_trigger_step()
        #alternative: self.trigger_step = random.uniform(2000, 5000)

    @staticmethod
    def validate_cfg(cfg) -> Tuple[bool, str]:
        if "offline" not in cfg:
            return False, "Missing 'offline' key in SITLogic config"
        trigger = cfg.get("trigger", SITLogic.DEFAULT_TRIGGER)
        trigger_variance = cfg.get("trigger_var", SITLogic.DEFAULT_TRIGGER)
        if trigger == SITLogic.DEFAULT_TRIGGER and trigger_variance <= 0:
            return False, "SITLogic trigger_var must be > 0 for stochastic trigger timing"

        return True, ""

    def _compute_trigger_step(self) -> int:
        # A configured deterministic trigger has no stochastic law to bias.
        if self.trigger != self.DEFAULT_TRIGGER:
            return round(self.trigger)

        # Base (target) and proposal distributions.
        trig_mean_base = self.trigger_mean
        trig_var_base = self.trigger_var
        trig_mean_bias = self.bias_cfg.get("trigger_mean", trig_mean_base)
        trig_var_bias = self.bias_cfg.get("trigger_var", trig_var_base)

        # Apply additive/multiplicative deltas if provided to bias the proposal,
        # while leaving the base distribution unchanged for weighting.
        trig_mean_bias += self.bias_cfg.get("trigger_mean_add", 0.0)
        trig_var_bias *= self.bias_cfg.get("trigger_var_scale", 1.0)

        # use sampler if provided
        if self.sampler:
            if trig_var_base <= 0 or trig_var_bias <= 0:
                raise SimulationConfigurationError(
                    "Importance-sampled SIT trigger timing requires positive base "
                    "and proposal trigger_var values."
                )
            audit_configs = self.bias_cfg.get("audit_proposals", {}) or {}
            audit_ids = tuple(getattr(self.sampler, "audit_proposal_ids", ()))
            unknown = sorted(
                set(audit_configs) - set(audit_ids)
            )
            if unknown:
                raise ValueError(f"Unknown fault audit proposal IDs: {unknown}")
            audit_params = {}
            for proposal_id in audit_ids:
                values = dict(audit_configs.get(proposal_id, {}))
                mean = float(values.get("trigger_mean", trig_mean_base))
                sigma = float(values.get("trigger_var", trig_var_base))
                mean += float(values.get("trigger_mean_add", 0.0))
                sigma *= float(values.get("trigger_var_scale", 1.0))
                audit_params[proposal_id] = (mean, sigma)
            sampler_args = {
                "mu_base": trig_mean_base,
                "sigma_base": trig_var_base,
                "mu_bias": trig_mean_bias,
                "sigma_bias": trig_var_bias,
                "component": "fault.trigger",
            }
            if audit_ids:
                sampler_args["audit_params_bias"] = audit_params
            return round(self.sampler.normal(**sampler_args))

        return round(random.normalvariate(trig_mean_base, trig_var_base))

    def step(self, sim_time) -> ComEvent:
        """

        :param sim_time: the time in milli from simulation start
        :return:
        """
        if self.online:
            if not self.triggered:
                if sim_time >= self.trigger_step:
                    self.triggered = True
                    self.online = False
                    self.start_offline_time = sim_time
            return ComEvent.ONLINE
        else:
            diff = sim_time - self.start_offline_time

            if diff > self.time_to_stay_offline:
                self.online = True
                #print("<DEBUG SIT> new code offline end")
                return ComEvent.ONLINE

            return ComEvent.OFFLINE
