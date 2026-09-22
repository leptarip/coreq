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
#
#
#
#
#
#
import shapely

from source.simulation_environment.agents.critical_region.critical_region import CriticalRegion
from source.simulation_environment.agents.critical_region.prediction_base import PredictionBase


class EgoPrediction(PredictionBase):

    def __init__(self,
                 reference_speed: float,
                 max_acc:float,
                 max_break:float,
                 vehicle_len: float,
                 critical_region: CriticalRegion,
                 ):
        """

        :param reference_speed: in [km/h]
        :param max_break: in [m/s]
        :param vehicle_len: in [m]
        :param critical_region:
        """
        super().__init__(reference_speed, max_break, vehicle_len, critical_region)
        self.max_acc = max_acc
        self.current_relative_pos = CriticalRegion.Position.UNKNOWN
        self.dist_to_cr = -1
        self.time_to_cr = -1
        self.time_to_leave_cr = -1



    def _get_current_relative_pos(self, front: shapely.Point):
        self.d_front, self.d_rear = self._project_to_path(front, self.len, 0)
        return self._get_relative_position(self.d_front, self.d_rear)

    def predict_go_relative_position(self, vel: float, time: float, front: shapely.Point):
        d_go = self.predict_go_displacement(vel, self.max_acc, time)
        d_go_front, d_go_rear = self._project_to_path(front, self.len, d_go)
        # check the position when using the current speed and target_speed for prediction
        return self._get_relative_position(d_go_front, d_go_rear)

    def predict_break_relative_position(self, vel:float , time: float, front: shapely.Point):
        # check the BREAK displacement
        d_break = self.predict_break_displacement(vel, time)
        d_break_front, d_break_rear = self._project_to_path(front, self.len, d_break)
        # check the position if we start to break
        return self._get_relative_position(d_break_front, d_break_rear)

    def reset(self):
        super().reset()
        self.current_relative_pos = CriticalRegion.Position.UNKNOWN
        self.dist_to_cr = -1
        self.time_to_cr = -1
        self.time_to_leave_cr = -1


    def predict(self, current_vel:float, front: shapely.Point):
        self.current_relative_pos = self._get_current_relative_pos(front)
        self.time_to_leave_cr =self.get_time_to_leave_cr(self.current_relative_pos, current_vel, self.max_acc)
        self.time_to_cr = self.get_time_to_cr(self.current_relative_pos, current_vel, self.max_acc)
        self.dist_to_cr = self.get_dist_to_cr(self.current_relative_pos)