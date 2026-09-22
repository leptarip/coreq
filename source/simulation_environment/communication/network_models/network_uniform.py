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
from typing import Tuple, Optional

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

class NetworkLogicUniform(NetworkLogicBase):

    def __init__(self, cfg:dict, sampler: Optional[ImportanceSampler] = None, bias_cfg: Optional[dict] = None):
        super().__init__()
        require_sampler_for_bias(sampler, bias_cfg, "Uniform network")
        self.drop_rate = cfg["packet_drop_rate"] # (a float between 0.0 and 1.0)
        self.delay = cfg["delay"]
        self.jitter = cfg["jitter"]
        self.sampler = sampler

        bias = bias_cfg or cfg
        self.bias_window = parse_bias_window(bias_cfg)
        self._window_successes = 0
        self.drop_rate_bias = bias.get("packet_drop_rate", self.drop_rate)
        self.delay_bias = bias.get("delay", self.delay)
        self.jitter_bias = bias.get("jitter", self.jitter)

    @staticmethod
    def validate_cfg(cfg:dict)-> Tuple[bool, str]:
        header = "NetworkUniform -->"

        if "packet_drop_rate" not in cfg or "delay" not in cfg or "jitter" not in cfg:
            return False, f"{header} Missing parameters"

        if cfg["delay"] < 0 or cfg["jitter"] < 0:
            return False, f"{header} Delay and Jitter must be positive"

        if not (0<=cfg["packet_drop_rate"]<=1):
            return False, f"{header} Packet drop rate must be between 0 and 1"

        return True, ""

    def compute_De2e(self, sim_time=None) -> int:
        if self.sampler is None:
            val = self.delay + random.uniform(0, self.jitter)
            return round(val)

        active = bias_window_active(self.bias_window, sim_time, self._window_successes)
        sample = self.sampler.uniform(
            a_base=self.delay,
            b_base=self.delay + self.jitter,
            a_bias=self.delay_bias if active else self.delay,
            b_bias=(self.delay_bias + self.jitter_bias) if active else (self.delay + self.jitter),
            component="network.delay.window" if self.bias_window is not None and active else "network.delay",
        )
        return round(sample)

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
        dropped = self.sampler.bernoulli(
            p_base=self.drop_rate,
            p_bias=self.drop_rate_bias if active else self.drop_rate,
            component=(
                "network.packet_drop.window"
                if self.bias_window is not None and active
                else "network.packet_drop"
            ),
        )
        if not dropped and bias_window_contains_time(self.bias_window, sim_time):
            self._window_successes += 1
        return dropped
