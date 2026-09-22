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

from source.simulation_environment.obps.measurement import Measurement
from source.simulation_environment.obps.sensor_prob import SensorProb


class SensorMsgContent:
    class MetaData:
        def __init__(self, fov_region, sensor_prob: SensorProb):
            self.fov_region = fov_region
            self.sensor_prob = sensor_prob

    def __init__(self, meta_data: MetaData | None, measurement: Measurement):
        self.meta_data = meta_data
        self.measurement = measurement


class SensorMsg:
    def __init__(self,
                 sender: str,
                 msg_id,
                 time_stamp,
                 content: SensorMsgContent
                 ):
        self.sender: str = sender
        self.id = msg_id  # message id, should be unique
        self.payload: SensorMsgContent = content
        self.time_stamp = time_stamp  # the instance of time when the message is generated
