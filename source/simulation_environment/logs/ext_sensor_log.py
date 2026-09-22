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
from source.simulation_environment.obps.measurement import Measurement


class ExtSensorLogItem:
    def __init__(self, time_stamp, sender, msg_id, result, payload: Measurement, reg_name: str):
        self.t = time_stamp
        self.result = result
        self.sender = sender
        self.reg_name = reg_name
        self.id = msg_id
        self.payload = payload

