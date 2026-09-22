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
from __future__ import annotations

from typing import List

from source.simulation_environment.obps.measurement import Measurement
from source.simulation_environment.obps.sensor_prob import SensorProb


class SensorResult:

    DETECTION = 1
    GHOST = 2
    WRONG = 3
    MISS = 4
    NO_DETECTION = 5

    def __init__(self,
                 result: int,
                 measurement: Measurement | None,
                 sensor_prob: SensorProb | None,
                 fov_region_name: str | None,
                 fov_region: List[()] | None):
        self.result = result
        self.measurement = measurement
        self.sensor_prob = sensor_prob
        self.fov_region = fov_region
        self.fov_region_name = fov_region_name

    def result_to_string(self) -> str:
        if self.result == SensorResult.DETECTION:
            return "det"
        if self.result == SensorResult.GHOST:
            return "ghost"
        if self.result == SensorResult.WRONG:
            return "wrong"
        if self.result == SensorResult.MISS:
            return "miss"
        if self.result == SensorResult.NO_DETECTION:
            return "no_det"
        return "unknown"

