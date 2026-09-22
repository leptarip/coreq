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
"""



"""
from enum import IntEnum


class TargetAssumption:
    class TargeType(IntEnum):
        VEHICLE = 0
        PEDESTRIAN = 1

    def __init__(self, max_speed:float, max_acc: float, max_break:float, agent_type: TargeType, target_len: float):
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.max_break = max_break
        self.agent_type = agent_type
        self.len = target_len