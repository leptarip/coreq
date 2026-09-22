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
from source.simulation_environment.agents.target_assumptions import TargetAssumption


class TargetPrediction(PredictionBase):

    def __init__(self,
                 target_id,
                 assumption: TargetAssumption,
                 critical_region: CriticalRegion
                 ):
        """

        :param target_id:
        :param assumption:
        :param critical_region:
        """
        super().__init__(assumption.max_speed, assumption.max_break, assumption.len, critical_region)
        self.target_id = target_id
        self.max_acc = assumption.max_acc
        self.time_to_cr = -1
        self.time_to_leave_cr = -1
        self.current_pred_relative_pos = CriticalRegion.Position.UNKNOWN


    def _get_predicted_aoi_displacement(self, aoi_s: float, current_vel: float, current_accel: float):
        """

          :param aoi_s: age of information in seconds
          :param current_vel:
          :param current_accel:
          :return:
        """
        if self.at_max_speed(current_vel):
            return self.max_speed * aoi_s
        else:
            t_x = (self.max_speed - current_vel) / current_accel
            if aoi_s - t_x >= 0:
                d_1 = current_vel * t_x + 0.5 * current_accel * t_x ** 2
                d_2 = self.max_speed * (aoi_s - t_x)
                d = d_1 + d_2
                return d
            else:
                d = current_vel * aoi_s + 0.5 * current_accel * aoi_s ** 2
                return d


    def _get_predicted_current_relative_position(self,
                                        aoi_s: float,
                                        current_vel: float,
                                        current_accel: float,
                                        front: shapely.Point,
                                        vehicle_len: float = None):
        """

        """
        # Compensate for the AoI by calculating the displacement that the target makes
        # in delta_t = (current_t - aoi)
        # Consider that the target is at aoi_comp_pos = (observed position + compensated displacement)
        if vehicle_len is None:
            vehicle_len = self.len

        displacement = self._get_predicted_aoi_displacement(aoi_s, current_vel, current_accel)

        self.d_front, self.d_rear = self._project_to_path(front, vehicle_len, displacement)

        return self._get_relative_position(self.d_front, self.d_rear)

    def reset(self):
        super().reset()
        self.current_pred_relative_pos = CriticalRegion.Position.UNKNOWN
        self.dist_to_cr = -1
        self.time_to_cr = -1
        self.time_to_leave_cr = -1


    def predict(self, aoi_s: float, vel:float, acc:float, target_length: float, front: shapely.Point):
        self.current_pred_relative_pos = self._get_predicted_current_relative_position(aoi_s, vel, acc, front, target_length)
        self.time_to_leave_cr =self.get_time_to_leave_cr(self.current_pred_relative_pos, vel, acc)
        self.time_to_cr = self.get_time_to_cr(self.current_pred_relative_pos, vel, acc)
        self.dist_to_cr = self.get_dist_to_cr(self.current_pred_relative_pos)


