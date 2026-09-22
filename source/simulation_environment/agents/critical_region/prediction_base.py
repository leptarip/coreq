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
import math
import shapely

from source.simulation_environment.agents.critical_region.critical_region import CriticalRegion
import source.simulation_environment.vehicle_constants as constants


class PredictionBase:
    NO_TIME_TO_CR = -1

    def __init__(self,
                 reference_speed: float,
                 max_break,
                 vehicle_len: float,
                 critical_region: CriticalRegion,
                 ):
        """

        :param reference_speed: in [km/h]
        :param max_break: in [m/s]
        :param vehicle_len: in [m]
        :param critical_region:
        """
        self.max_speed = reference_speed / constants.MITER_KM_RATIO
        self.max_break = max_break
        self.len = vehicle_len
        self.cr = critical_region
        self.dist_to_cr = -1
        self.d_front = -1
        self.d_rear = -1
        self.eps = 0.01
        self.v_delta = 0.1


    def reset(self):
        self.dist_to_cr = -1
        self.d_front = -1
        self.d_rear = -1


    def at_max_speed(self, target_vel)-> bool:
        cond1 = math.fabs(target_vel - self.max_speed) < self.v_delta
        cond2 = target_vel > self.max_speed
        return cond1 or cond2

    def _get_time_to(self, vel, acc, distance):
        # check if we are at max speed
        if self.at_max_speed(vel):
            return distance / self.max_speed

        # Treat extremely small velocities as zero
        if abs(vel) < self.eps:
            vel = 0.0

        # if we are not at max speed we should calculate the time piece wise
        d1 = ((self.max_speed ** 2) - (vel ** 2)) / (2 * acc)
        if distance > d1:
            # accelerated motion
            t1 = (self.max_speed - vel) / acc
            # constant motion
            d2 = distance - d1
            t2 = d2 / self.max_speed
            return t1 + t2
        else:
            # only accelerated motion, that means that while moving up to distance we never reach max speed
            # discriminant V*v - 4*0.5*a*(-distance) -> V*v + 4*0.5*a*(distance)
            discriminant = vel ** 2 - 2 * acc * (-distance)
            t = (-vel + math.sqrt(discriminant)) / acc
            return t

    def predict_go_displacement(self, velocity: float, accel: float, t: float):
        """
        Predict where ego will be in the given time t
        :param accel: current acceleration
        :param velocity: the current velocity
        :param t: time to perform the manoeuvre
        :return: location after time t, distance driven in time t
        """

        if velocity < self.max_speed and accel > 0:
            t_x = (self.max_speed - velocity) / accel
            if t - t_x >= 0:
                d_1 = velocity * t_x + 0.5 * accel * t_x ** 2
                d_2 = self.max_speed * (t - t_x)
                d = d_1 + d_2
            else:
                d = velocity * t + 0.5 * accel * t ** 2
        else:
            # steady state velocity
            d = velocity * t

        return d

    def predict_break_displacement(self, velocity: float, t: float):
        """

        :param velocity: the current velocity
        :param t: time to perform the manoeuvre
        :return: location after time t, distance driven in time t
        """

        d = velocity * t - 0.5 * self.max_break * (t ** 2)
        d = max(0.0, d)

        return d

    def get_time_to_leave_cr(self, relative_pos: CriticalRegion.Position, current_vel: float, accel: float) -> float:
        t = self.NO_TIME_TO_CR

        if relative_pos == CriticalRegion.Position.BEFORE_CR or relative_pos == CriticalRegion.Position.INSIDE_CR:
            # get distance to leave the CR
            distance = math.fabs(self.cr.cf_orig_d - self.d_rear)
            t = self._get_time_to(current_vel, accel, distance)

        return t

    def get_time_to_cr(self,
                       relative_pos: CriticalRegion.Position,
                       current_vel: float,
                       accel: float):
        """


        :param relative_pos:
        :param current_vel:
        :param accel:
        :return:
        """
        # This time only exists if we are before the region
        target_time = self.NO_TIME_TO_CR

        if relative_pos == CriticalRegion.Position.BEFORE_CR:
            # get distance to enter the CR
            # we are before the CN point!
            distance = math.fabs(self.cr.cn_orig_d - self.d_front)
            target_time = self._get_time_to(current_vel, accel, distance)

        return target_time

    def get_dist_to_cr(self, relative_pos: CriticalRegion.Position):
        if relative_pos == CriticalRegion.Position.UNKNOWN:
            return -1

        if relative_pos == CriticalRegion.Position.AFTER_CR or relative_pos == CriticalRegion.Position.INSIDE_CR:
            return 0

        if relative_pos == CriticalRegion.Position.BEFORE_CR:
            return max(0, self.cr.cn_orig_d - self.d_front)

    def _get_relative_position(self, front_d, rear_d):
        if front_d < self.cr.cn_orig_d:
            return CriticalRegion.Position.BEFORE_CR

        if rear_d > self.cr.cf_orig_d:
            return CriticalRegion.Position.AFTER_CR

        if self.cr.cn_orig_d <= front_d <= self.cr.cf_orig_d or self.cr.cn_orig_d <= rear_d <= self.cr.cf_orig_d:
            return CriticalRegion.Position.INSIDE_CR


    def _project_to_path(self, front: shapely.Point, length: float, displacement: float):
        # project the detected front on the critical path and find its distance
        d_front = shapely.line_locate_point(self.cr.cr_path, front) + displacement

        # find the rear point distance relative to the front
        d_rear = d_front - length

        return d_front, d_rear

