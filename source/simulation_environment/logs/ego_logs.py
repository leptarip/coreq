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
import math
from typing import List

import carla
import numpy as np
from source.simulation_environment.obps.message import SensorMsg


class CollisionLogItem:
    def __init__(self, t, v, impulse_x, impulse_y, impulse_z):
        self.t = t
        self.value = v
        self.impulse_x = impulse_x
        self.impulse_y = impulse_y
        self.impulse_z = impulse_z


class ImuLogItem:
    def __init__(self, a, a_x, a_y, a_z):
        self.a = a
        self.a_x = a_x
        self.a_y = a_y
        self.a_z = a_z


class PerceptionLogItem:
    def __init__(self, t, id_target, dist_to_target):
        self.t = t
        self.id_target = id_target
        self.dist_to_target = dist_to_target



class DataLogItem:
    def __init__(self, t,
                 ac, ac_x, ac_y, ac_z,
                 vel, v_x, v_y,
                 pos_x, pos_y, fw_x, fw_y, rg_x, rg_y,
                 dist_to_obs,
                 v_selected,
                 v_obs,
                 ):
        #self.t = t
        self.ac = ac
        self.ac_x = ac_x
        self.ac_y = ac_y
        self.ac_z = ac_z
        self.vel = vel
        self.v_x = v_x
        self.v_y = v_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.fw_x = fw_x
        self.fw_y = fw_y
        self.rg_x = rg_x
        self.rg_y = rg_y
        self.dist_to_obs = dist_to_obs
        self.v_selected = v_selected
        self.v_obs = v_obs


class MsgLogItem:
    def __init__(self, sim_time, sender: str, msg_id: str):
        self.t = sim_time
        self.sender = sender
        self.id = msg_id

def log_ego_data(t,
                 a_world,
                 vel_world,
                 trans: carla.Transform,
                 dist_to_obs: float,
                 v_selected: float,
                 v_obs: float) -> DataLogItem:

        """

        :param v_obs:
        :param v_selected:
        :param t: time
        :param a_world:  carla.Vector3 acceleration in word reference
        :param vel_world: carla.Vector3 velocity in word reference
        :param trans: carla.Transform
        :param dist_to_obs: distance to next obstacle in meters
        :return:
        """
        fw = trans.get_forward_vector()
        rg = trans.get_right_vector()
        up = trans.get_up_vector()
        m = np.zeros((3, 3))
        m[0, :] = np.array([fw.x, fw.y, fw.z])
        m[1, :] = np.array([rg.x, rg.y, rg.z])
        m[2, :] = np.array([up.x, up.y, up.z])

        v_vector = np.array([vel_world.x, vel_world.y, vel_world.z]).T
        v_ego = np.dot(m, v_vector)

        a_vector = np.array([a_world.x, a_world.y, a_world.z]).T
        a_ego = np.dot(m, a_vector)

        vel = math.sqrt(v_ego[0] ** 2 + v_ego[1] ** 2)
        ac = math.sqrt(a_ego[0] ** 2 + a_ego[1] ** 2)

        data_item = DataLogItem(
            t,
            ac,
            a_ego[0],
            a_ego[1],
            a_ego[2],
            vel,
            v_ego[0],
            v_ego[1],
            trans.location.x,
            trans.location.y,
            fw.x,
            fw.y,
            rg.x,
            rg.y,
            dist_to_obs,
            v_selected,
            v_obs
        )

        return data_item


def log_collisions(point):
    """

    :param point: collision
    """
    col_data = CollisionLogItem(point[0], point[1], point[2], point[3], point[4])
    return col_data
    # rd.save_stream_ego_msg(self.experiment.col_msg_name, msg_log_data)


def log_ext_sensor_messages(sim_time, msgs: List[SensorMsg]) -> List[MsgLogItem]:
    app = [MsgLogItem(sim_time, msg.sender, msg.id) for msg in msgs]
    return app
    #rd.save_stream_ego_msg(self.experiment.stream_obps_msg_name, msg_log_data)

