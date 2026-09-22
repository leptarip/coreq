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
import random
import math
import source.simulation_environment.vehicle_constants as constants
import source.simulation_environment.logs.printl

printl = source.simulation_environment.logs.printl.PrintL("LVelCtrl", enabled=True)

class LongVelController:
    MARGIN_DISTANCE = 0.8  # [m]
    CAR_FRONT_OFFSET = 0  # if the point from where to calculate is not the front bumper[m]

    def __init__(self,
                 a_max_acc=constants.MAX_ACC,
                 a_max_deceleration=constants.MAX_BRAKE,
                 a_min_deceleration=constants.MIN_BRAKE,
                 min_response_time=constants.MIN_RESPONSE_TIME,
                 max_response_time=constants.MAX_RESPONSE_TIME):

        self.a_max_acc = a_max_acc
        self.a_max_deceleration = a_max_deceleration
        self.a_min_deceleration = a_min_deceleration
        self.min_response_time = min_response_time
        self.max_response_time = max_response_time

    def get_min_distance(self, v_r, v_f=0):
        p = self.get_response_time()

        d_min = v_r * p + 0.5 * self.a_max_acc * pow(p, 2) + 0.5 * pow(v_r + p * self.a_max_acc, 2) / self.a_min_deceleration

        if v_f != 0:
            d_min = d_min - 0.5 * pow(v_f, 2) / constants.MAX_BRAKE

        d = max(d_min, 0)

        return d

    def get_response_time(self):
        return random.uniform(self.min_response_time, self.max_response_time)

    def get_a_brake(self):
        return self.a_max_deceleration

    def compute_velocity(self,
                         d_to_obs: float,
                         car_front_offset=CAR_FRONT_OFFSET,
                         margin_distance=MARGIN_DISTANCE):
        """
        equation:
        d = d_to_conflict + v_r * p + 0.5 * self.a_max_acc * pow(p, 2) + 0.5 * pow(v_p_max, 2) / self.a_min_brake
        :param car_front_offset:
        :param d_to_obs:
        :param margin_distance: amount in meters to stop before the object
        :return: velocity [m/s], unfeasible velocity [m/s], if it is feasible to break
        """
        p = self.get_response_time()
        # max velocity reachable during response time p
        # v_p_max = pow(v_r + self.a_max_acc * p,2)
        M = car_front_offset + margin_distance

        a = 1
        b = 2 * self.a_min_deceleration * p + 2 * self.a_max_acc * p
        c = 2 * self.a_min_deceleration * (-d_to_obs + M + 0.5 * self.a_max_acc * pow(p, 2)) + pow(p, 2) * pow(self.a_max_acc,
                                                                                                               2)
        dis = pow(b, 2) - 4 * a * c

        feasible = True

        if dis >= 0:
            v_1 = (-b + math.sqrt(dis)) / 2
        else:
            v_1 = -1
            printl.to_print(message=f'negative discriminant, no space to stop with given speed, dist={d_to_obs}')

        v_unfeasible = v_1

        if v_1 < 0:
            feasible = False
            v_1 = 0

        return v_1, v_unfeasible, feasible
