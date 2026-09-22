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

from source.simulation_environment.agents.target_assumptions import TargetAssumption


class TacticalCfg:

    def __init__(self):
        """
        """
        self.vehicles_assumptions = list()
        self.pedestrians_assumptions = list()

    def add_assumption(self, assumption:TargetAssumption):

        if assumption.agent_type == assumption.TargeType.VEHICLE:
            self.vehicles_assumptions.append(assumption)

        if assumption.agent_type == assumption.TargeType.PEDESTRIAN:
            self.pedestrians_assumptions.append(assumption)

    def get_vehicle_assumption(self)->TargetAssumption:
        return self.vehicles_assumptions[-1]

    def get_pedestrian_assumption(self)->TargetAssumption:
        return self.pedestrians_assumptions[-1]



