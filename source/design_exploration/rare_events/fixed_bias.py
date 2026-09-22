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

"""Declarative fixed proposal used after the Phase-3 pilot is frozen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class FixedBiasPlan:
    """Proposal changes applied only to simulator randomness, never the design."""

    name: str = "phase3_tail_push_v1"
    sensor_miss_scale: float = 4.0
    sensor_unc_pos_add: float = 2.0
    sensor_unc_vel_add: float = 2.0
    network_delay_min_add: float = 0.0
    network_delay_avg_add: float = 150.0
    network_jitter_scale: float = 4.0
    network_drop_add: float = 0.40
    fault_trigger_mean_add: float = 0.0
    fault_trigger_var_scale: float = 1.0
    defensive_mixture_probability: float = 0.2
    sensor_window_start_ms: Optional[int] = None
    sensor_window_end_ms: Optional[int] = None
    sensor_window_until_successes: Optional[int] = None
    network_window_start_ms: Optional[int] = None
    network_window_end_ms: Optional[int] = None
    network_window_until_successes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sensor_miss_scale <= 0.0:
            raise ValueError("Sensor miss scale must be positive.")
        if self.network_jitter_scale <= 0.0:
            raise ValueError("Network jitter scale must be positive.")
        if self.fault_trigger_var_scale <= 0.0:
            raise ValueError("Fault trigger variance scale must be positive.")
        if not 0.0 < self.defensive_mixture_probability < 1.0:
            raise ValueError("defensive_mixture_probability must be in (0, 1).")
        if self.network_delay_min_add != 0.0:
            raise ValueError(
                "Changing the minimum network delay loses nominal support; "
                "network_delay_min_add must be zero."
            )
        self._validate_window(
            self.sensor_window_start_ms, self.sensor_window_end_ms,
            self.sensor_window_until_successes, "sensor"
        )
        self._validate_window(
            self.network_window_start_ms, self.network_window_end_ms,
            self.network_window_until_successes, "network"
        )

    @staticmethod
    def _validate_window(
        start: Optional[int], end: Optional[int], until_successes: Optional[int], name: str
    ) -> None:
        if (start is None) != (end is None):
            raise ValueError(f"{name} proposal window requires both start and end.")
        if start is not None and (int(start) < 0 or int(end) < int(start)):
            raise ValueError(f"{name} proposal window requires 0 <= start <= end.")
        if until_successes is not None and (
            start is None or int(until_successes) < 1
        ):
            raise ValueError(
                f"{name} until-successes conditioning requires a window and positive count."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FixedBiasPlan":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown proposal fields: {unknown}")
        return cls(**dict(values))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_identity(self) -> bool:
        """Whether every proposal distribution equals its nominal counterpart."""
        return (
            self.sensor_miss_scale == 1.0
            and self.sensor_unc_pos_add == 0.0
            and self.sensor_unc_vel_add == 0.0
            and self.network_delay_min_add == 0.0
            and self.network_delay_avg_add == 0.0
            and self.network_jitter_scale == 1.0
            and self.network_drop_add == 0.0
            and self.fault_trigger_mean_add == 0.0
            and self.fault_trigger_var_scale == 1.0
        )

    def scaled(self, scale: float) -> "FixedBiasPlan":
        """Interpolate from the nominal proposal (scale=0) to this plan (scale=1)."""
        scale = float(scale)
        if scale < 0.0:
            raise ValueError("Proposal scale must be non-negative.")
        return FixedBiasPlan(
            name=f"{self.name}_scale_{scale:.6g}",
            sensor_miss_scale=1.0 + (self.sensor_miss_scale - 1.0) * scale,
            sensor_unc_pos_add=self.sensor_unc_pos_add * scale,
            sensor_unc_vel_add=self.sensor_unc_vel_add * scale,
            network_delay_min_add=self.network_delay_min_add * scale,
            network_delay_avg_add=self.network_delay_avg_add * scale,
            network_jitter_scale=1.0 + (self.network_jitter_scale - 1.0) * scale,
            network_drop_add=self.network_drop_add * scale,
            fault_trigger_mean_add=self.fault_trigger_mean_add * scale,
            fault_trigger_var_scale=1.0 + (self.fault_trigger_var_scale - 1.0) * scale,
            defensive_mixture_probability=self.defensive_mixture_probability,
            sensor_window_start_ms=self.sensor_window_start_ms,
            sensor_window_end_ms=self.sensor_window_end_ms,
            sensor_window_until_successes=self.sensor_window_until_successes,
            network_window_start_ms=self.network_window_start_ms,
            network_window_end_ms=self.network_window_end_ms,
            network_window_until_successes=self.network_window_until_successes,
        )

    def to_simulator_config(
        self,
        design: Mapping[str, Any],
        *,
        d_safe: float,
        near_miss_threshold: float,
        seed: int,
    ) -> Dict[str, Any]:
        """Return the existing simulator's ``rare_event`` configuration block."""
        drop_base = float(design["packet_drop_rate"])
        drop_bias = drop_base + self.network_drop_add
        if drop_base > 0.0 and drop_bias <= 0.0:
            raise ValueError("Biased packet-drop probability loses nominal support.")
        if drop_base < 1.0 and drop_bias >= 1.0:
            raise ValueError("Biased packet-drop probability loses nominal support.")

        delay_min = float(design["network.delay_min"])
        delay_avg = float(design["network.delay_avg"])
        jitter = float(design["network.jit"])
        delay_min_bias = delay_min + self.network_delay_min_add
        delay_avg_bias = delay_avg + self.network_delay_avg_add
        jitter_bias = jitter * self.network_jitter_scale
        if delay_avg_bias <= delay_min_bias:
            raise ValueError("Biased average delay must exceed biased minimum delay.")
        if jitter_bias <= 0.0:
            raise ValueError("Biased network jitter must be positive.")

        sensor_bias = {
            "miss_scale": self.sensor_miss_scale,
            "unc_p_add": self.sensor_unc_pos_add,
            "unc_v_add": self.sensor_unc_vel_add,
        }
        if self.sensor_window_start_ms is not None:
            sensor_bias["active_window"] = {
                "start_ms": int(self.sensor_window_start_ms),
                "end_ms": int(self.sensor_window_end_ms),
            }
            if self.sensor_window_until_successes is not None:
                sensor_bias["active_window"]["until_successes"] = int(
                    self.sensor_window_until_successes
                )
        network_bias = {
            "delay_min": delay_min_bias,
            "delay_avg": delay_avg_bias,
            "jitter": jitter_bias,
            "packet_drop_rate": drop_bias,
        }
        if self.network_window_start_ms is not None:
            network_bias["active_window"] = {
                "start_ms": int(self.network_window_start_ms),
                "end_ms": int(self.network_window_end_ms),
            }
            if self.network_window_until_successes is not None:
                network_bias["active_window"]["until_successes"] = int(
                    self.network_window_until_successes
                )

        return {
            "enabled": True,
            "seed": int(seed),
            "plan_name": self.name,
            "defensive_mixture_probability": self.defensive_mixture_probability,
            "event": {"metric": "min_d", "op": "<=", "threshold": float(d_safe)},
            "near_miss": {
                "metric": "min_d",
                "op": "<=",
                "threshold": float(near_miss_threshold),
            },
            "sensor_bias": sensor_bias,
            "network_bias": network_bias,
            "fault_bias": {
                "trigger_mean_add": self.fault_trigger_mean_add,
                "trigger_var_scale": self.fault_trigger_var_scale,
            },
        }
