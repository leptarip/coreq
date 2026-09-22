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
from json import JSONEncoder

import redis
import json
import plotly

from source.simulation_environment.logs.com_logs import ComLogItem
from source.simulation_environment.logs.tactical_log import TacticalLog
from source.simulation_environment.logs.ego_logs import DataLogItem, ImuLogItem, MsgLogItem, CollisionLogItem, PerceptionLogItem
from source.simulation_environment.logs.ext_sensor_log import ExtSensorLogItem
from source.simulation_environment.logs.scenario_log import SimScenarioItem
from source.simulation_environment.scenario import scenario_roles

K_EXT_SENSOR_MSG = "EXT_SENS_MSG"
K_COM = "COM"
K_SCENARIO_TERMINATION = "SC_TERM"

K_EGO_DATA = "EGO"
K_EGO_MSG = "EGO_MSG"
K_EGO_TACTICAL = "EGO_RSK"
K_EGO_COL = "EGO_COL"
K_EGO_IMU = "EGO_IMU"
K_EGO_PERCEPT = "EGO_PER"

EGO_KEYS = [K_EGO_DATA, K_EGO_MSG, K_EGO_TACTICAL, K_EGO_COL, K_EGO_IMU, K_EGO_PERCEPT]

K_ADVERSARY_DATA = "ADV"
K_ADVERSARY_MSG = "ADV_MSG"
K_ADVERSARY_TACTICAL = "ADV_RISK"
K_ADVERSARY_COL = "ADV_COL"
K_ADVERSARY_IMU = "ADV_IMU"
K_ADVERSARY_PERCEPT = "ADV_PER"

ADV_KEYS = [K_ADVERSARY_DATA, K_ADVERSARY_MSG, K_ADVERSARY_TACTICAL, K_ADVERSARY_COL, K_ADVERSARY_IMU, K_ADVERSARY_PERCEPT]

K_LOG = "LOG"
H_SET_EXP = "EXP"

redis_instance = redis.Redis(host='localhost', port=6379, db=0)

class SimRunKeys:
    def __init__(self, exp_name, role):
        self.exp_name = exp_name
        self.role = role

        if self.role == scenario_roles.ROLE_HERO:
            self.data_key = get_h_key(self.exp_name, K_EGO_DATA)
            self.msg_key = get_h_key(self.exp_name, K_EGO_MSG)
            self.tactical_key = get_h_key(self.exp_name, K_EGO_TACTICAL)
            self.collision_key = get_h_key(self.exp_name, K_EGO_COL)
            self.imu_key = get_h_key(self.exp_name, K_EGO_IMU)
            self.perception_key = get_h_key(self.exp_name, K_EGO_PERCEPT)
        else:
            self.data_key = get_h_key(self.exp_name, K_ADVERSARY_DATA)
            self.msg_key = get_h_key(self.exp_name, K_ADVERSARY_MSG)
            self.tactical_key = get_h_key(self.exp_name, K_ADVERSARY_TACTICAL)
            self.collision_key = get_h_key(self.exp_name, K_ADVERSARY_COL)
            self.imu_key = get_h_key(self.exp_name, K_ADVERSARY_IMU)
            self.perception_key = get_h_key(self.exp_name, K_ADVERSARY_PERCEPT)

        self.stream_data_name = get_stream_name(exp_name, self.data_key.decode())
        self.stream_obps_msg_name = get_stream_name(exp_name, self.msg_key.decode())
        self.stream_tactical_name = get_stream_name(exp_name, self.tactical_key.decode())
        self.stream_perception_name = get_stream_name(exp_name, self.perception_key.decode())
        self.stream_imu_name = get_stream_name(exp_name, self.imu_key.decode())
        self.stream_collision_name = get_stream_name(exp_name, self.collision_key.decode())

class SimpleEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def save_data(h_set_name, key, data, json_encoder=plotly.utils.PlotlyJSONEncoder):
    # Save the data in redis so that the Dash app, running on a separate
    # process, can read it
    try:
        redis_instance.hset(
            h_set_name,
            key,
            json.dumps(
                data,
                cls=json_encoder,
            )
        )
    except Exception as e:
        print("[REDIS] ERROR on saving data! {0}".format(e))


def save_stream_ego_data(stream: str, data: DataLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_ego_msg(stream: str, data: MsgLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_ego_imu(stream: str, data: ImuLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_ego_collision(stream: str, data: CollisionLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})

def save_stream_ego_perception(stream: str, data: PerceptionLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})

def save_stream_ego_tactical(stream: str, data: TacticalLog):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_com(stream: str, data: ComLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_ext_sensor(stream: str, data: ExtSensorLogItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_scenario_termination(stream: str, data: SimScenarioItem):
    value = json.dumps(data, cls=SimpleEncoder)
    redis_instance.xadd(name=stream, fields={"data": value})


def save_stream_log(stream: str, data: str):
    redis_instance.xadd(name=stream, fields={"data": data})


def get_stream_name(exp_name: str, key: str) -> str:
    return str(exp_name) + ":" + str(key)


def close_instance():
    redis_instance.close()


def set_exp_name(exp_name: str):
    redis_instance.set(H_SET_EXP, exp_name)

def get_key(key: str):
    return redis_instance.get(key)

def get_h_key(h_set: str, key: str):
    return redis_instance.hget(h_set, key)


def set_all_ego_keys(h_set: str, episode_num:int):
    for k in EGO_KEYS:
        value = f"{k}:{episode_num}"
        redis_instance.hset(h_set, k, value)

def set_all_adv_keys(h_set: str, episode_num:int):
    for k in ADV_KEYS:
        value = f"{k}:{episode_num}"
        redis_instance.hset(h_set, k, value)


def set_up_new_keys(h_set, episode_num: int):
    redis_instance.hset(h_set, K_LOG, f"{K_LOG}:{episode_num}")

    set_all_ego_keys(h_set, episode_num)
    set_all_adv_keys(h_set, episode_num)

    redis_instance.hset(h_set, K_COM, f"{K_COM}:{episode_num}")
    redis_instance.hset(h_set, K_EXT_SENSOR_MSG, f"{K_EXT_SENSOR_MSG}:{episode_num}")
    redis_instance.hset(h_set, K_SCENARIO_TERMINATION, f"{K_SCENARIO_TERMINATION}:{episode_num}")
