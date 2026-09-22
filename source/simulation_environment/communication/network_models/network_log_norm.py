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
import math
from typing import Optional, Tuple

import scipy.stats as stats
from source.simulation_environment.communication.network_models.network_logic_base import NetworkLogicBase
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    require_sampler_for_bias,
)
from source.simulation_environment.rare_events.window import (
    bias_window_active,
    bias_window_contains_time,
    parse_bias_window,
)

class NetworkLogicLogNorm(NetworkLogicBase):
    def __init__(self, cfg:dict, sampler: Optional[ImportanceSampler] = None, bias_cfg: Optional[dict] = None):
        super().__init__()
        require_sampler_for_bias(sampler, bias_cfg, "Log-normal network")
        self.sampler = sampler

        # Base parameters
        self.drop_rate = cfg["packet_drop_rate"] # (a float between 0.0 and 1.0)
        self.delay_min = cfg["delay_min"]
        self.delay_avg = cfg["delay_avg"]
        self.jitter = cfg["jitter"]

        # Bias parameters (fall back to base if not provided)
        bias = bias_cfg or cfg
        self.bias_window = parse_bias_window(bias_cfg)
        self._window_successes = 0
        self.drop_rate_bias = bias.get("packet_drop_rate", self.drop_rate)
        self.delay_min_bias = bias.get("delay_min", self.delay_min)
        self.delay_avg_bias = bias.get("delay_avg", self.delay_avg)
        self.jitter_bias = bias.get("jitter", self.jitter)

        # Derived log-normal params for base and bias distributions
        self.mu_base, self.sigma_base = self._fit_lognormal_params(self.delay_avg - self.delay_min, self.jitter)
        self.mu_bias, self.sigma_bias = self._fit_lognormal_params(self.delay_avg_bias - self.delay_min_bias,
                                                                   self.jitter_bias)

        audit_configs = bias.get("audit_proposals", {}) or {}
        audit_ids = (
            tuple(getattr(self.sampler, "audit_proposal_ids", ()))
            if self.sampler else ()
        )
        unknown = sorted(set(audit_configs) - set(audit_ids))
        if unknown:
            raise ValueError(f"Unknown network audit proposal IDs: {unknown}")
        self.audit_network_biases = {}
        for proposal_id in audit_ids:
            values = dict(audit_configs.get(proposal_id, {}))
            delay_min = float(values.get("delay_min", self.delay_min))
            delay_avg = float(values.get("delay_avg", self.delay_avg))
            jitter = float(values.get("jitter", self.jitter))
            mu, sigma = self._fit_lognormal_params(delay_avg - delay_min, jitter)
            self.audit_network_biases[proposal_id] = {
                "delay_min": delay_min,
                "mu": mu,
                "sigma": sigma,
                "packet_drop_rate": float(
                    values.get("packet_drop_rate", self.drop_rate)
                ),
                "window": parse_bias_window(values),
                "successes": 0,
            }

        # Frozen distribution for the unbiased path
        self.dist_unshifted = stats.lognorm(s=self.sigma_base, scale=math.exp(self.mu_base))

    @staticmethod
    def _fit_lognormal_params(mean_variable: float, std_variable: float) -> Tuple[float, float]:
        variance_variable = std_variable ** 2
        log_arg = variance_variable / (mean_variable ** 2) + 1
        variance_log = math.log(log_arg)
        sigma = math.sqrt(variance_log)
        mu = math.log(mean_variable) - variance_log / 2.0
        return mu, sigma

    @staticmethod
    def validate_cfg(cfg:dict)-> Tuple[bool, str]:
        header = "NetworkLogNorm -->"

        if ("packet_drop_rate" not in cfg or
                "delay_min" not in cfg or
                "delay_avg" not in cfg or
                "jitter" not in cfg):
            return False, f"{header} Missing parameters"

        if cfg["delay_min"] < 0 or cfg["delay_avg"] < 0 or cfg["jitter"] < 0:
            return False, f"{header} Parameters must be >0"

        if cfg["delay_avg"] - cfg["delay_min"] < 0:
            return False, f"{header} Delay_avg must be > delay_min"

        if not (0<=cfg["packet_drop_rate"]<=1):
            return False, f"{header} Packet drop rate must be between 0 and 1"

        return True, ""

    def compute_De2e(self, sim_time=None) -> int:
        """
        Draw one random samples (delays) from the distribution.

        Returns:
            float or np.ndarray: A single delay value (if size=1) or
                                 an array of delay values.
        """
        if self.sampler is None:
            variable_delay = self.dist_unshifted.rvs(size=None)
            total_delay = self.delay_min + variable_delay
            return round(total_delay)

        active = bias_window_active(self.bias_window, sim_time, self._window_successes)
        audit_params = {}
        for proposal_id, values in self.audit_network_biases.items():
            audit_active = bias_window_active(
                values["window"], sim_time, values["successes"]
            )
            audit_params[proposal_id] = (
                (values["delay_min"], values["mu"], values["sigma"])
                if audit_active
                else (self.delay_min, self.mu_base, self.sigma_base)
            )
        sampler_args = dict(
            shift_base=self.delay_min,
            mu_base=self.mu_base,
            sigma_base=self.sigma_base,
            shift_bias=self.delay_min_bias if active else self.delay_min,
            mu_bias=self.mu_bias if active else self.mu_base,
            sigma_bias=self.sigma_bias if active else self.sigma_base,
            component="network.delay.window" if self.bias_window is not None and active else "network.delay",
        )
        if self.audit_network_biases:
            sampler_args["audit_params_bias"] = audit_params
        total_delay = self.sampler.shifted_lognormal(**sampler_args)
        return round(total_delay)

    def should_drop_packet(self, sim_time=None) -> bool:
        """
        Determines whether to discard a packet based on a given probability.

        This function simulates a random event with a probability `p`. It's
        useful in network simulations for modeling packet loss.

        Returns:
            True if the packet should be discarded, False otherwise.

        """
        if self.drop_rate == 0 and (self.sampler is None or self.drop_rate_bias == 0):
            return False

        if self.sampler is None:
            return random.random() < self.drop_rate

        active = bias_window_active(self.bias_window, sim_time, self._window_successes)
        audit_probs = {
            proposal_id: (
                values["packet_drop_rate"]
                if bias_window_active(
                    values["window"], sim_time, values["successes"]
                )
                else self.drop_rate
            )
            for proposal_id, values in self.audit_network_biases.items()
        }
        sampler_args = dict(
            p_base=self.drop_rate,
            p_bias=self.drop_rate_bias if active else self.drop_rate,
            component=(
                "network.packet_drop.window"
                if self.bias_window is not None and active
                else "network.packet_drop"
            ),
        )
        if self.audit_network_biases:
            sampler_args["audit_probs_bias"] = audit_probs
        dropped = self.sampler.bernoulli(**sampler_args)
        if not dropped and bias_window_contains_time(self.bias_window, sim_time):
            self._window_successes += 1
        if not dropped:
            for values in self.audit_network_biases.values():
                if bias_window_contains_time(values["window"], sim_time):
                    values["successes"] += 1
        return dropped
