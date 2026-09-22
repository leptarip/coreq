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
from typing import Tuple


class VehicleCfg:

    DEFAULT_PREDICTION_WINDOW = 2.0

    def __init__(self,
                 role_name: str,
                 max_control_throttle: float,
                 max_control_break: float,
                 max_acceleration: float,
                 max_deceleration: float,
                 target_speed: float,
                 initial_speed: float,
                 sensor_range: int,
                 sensor_resolution: int,
                 sensor_fov: int,
                 prediction_window: float,
                 decision_period: int,
                 stream=None):
        """

        :param role_name:
        :param max_control_throttle:
        :param max_control_break:
        :param max_acceleration:
        :param max_deceleration:
        :param target_speed:
        :param sensor_range:
        :param sensor_resolution:
        :param sensor_fov:
        :param prediction_window: how far into the future we predict the object motion
        :param decision_period: in milliseconds. The whole AV stack will be executed every decision_period milliseconds.
        :param initial_speed:
        :return:
        """
        self.role_name = role_name
        self.max_throttle = max_control_throttle
        self.max_break = max_control_break
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.target_speed = target_speed
        self.sensor_range = sensor_range
        self.sensor_resolution = sensor_resolution
        self.sensor_fov = sensor_fov
        self.prediction_window = prediction_window
        self.decision_period = decision_period
        self.initial_speed = initial_speed
        self.stream = stream

    @staticmethod
    def validate_cfg(cfg: dict) -> Tuple[bool, str]:

        if ("target_v" not in cfg or
            "initial_v" not in cfg or
            "perception" not in cfg or
            "decision_period" not in cfg or
            "role_name" not in cfg
        ):
            return False, "Missing required parameters"

        if "fov" not in cfg["perception"]:
            return False, "Missing fov parameter"

        if ("degrees" not in cfg["perception"]["fov"] or
            "ray_step" not in cfg["perception"]["fov"] or
            "range" not in cfg["perception"]["fov"]):
            return False, "Missing fov parameters"

        if "prediction_window" not in cfg["perception"]:
            cfg["perception"]["prediction_window"] = VehicleCfg.DEFAULT_PREDICTION_WINDOW

        return True, ""

