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

import source.simulation_environment.logs.redis_data as rd
import json


class CommData:
    def __init__(self):
        self.online = list()
        self.dropped_packets = list()

    def add(self, j_data):
        self.online.append(j_data["online"])
        self.dropped_packets.extend(j_data["dropped_packets"])


class ExtSensorData:
    def __init__(self):
        self.data = {}

    def add(self, j_data):
        if j_data["sender"] not in self.data:
            self.data[j_data["sender"]] = list()
        self.data[j_data["sender"]].append(
            {"t": j_data["t"],
             "id": j_data["id"],
             "reg": j_data["reg_name"],
             "result": j_data["result"],
             "payload": j_data["payload"]
             })

class PerceptionData:
    def __init__(self):
        self.blocks = dict()

    def add(self, j_data):
        id_target = j_data.get("id_target")
        if id_target not in self.blocks:
            self.blocks[id_target] = {
                "t" : [],
                "dist_to_target" : []
            }
        # Append top level value
        self.blocks[id_target]["t"].append(j_data.get("t"))
        self.blocks[id_target]["dist_to_target"].append(j_data.get("dist_to_target"))

class TacticalData:
    def __init__(self):
        self.v_tactical = list()
        self.blocks = dict()

    def add(self, j_data):
        self.v_tactical.append(j_data["v_tactical"])
        if any(j_data["blocks"]):
            for key, value in j_data["blocks"].items():
                # Initialize the structure for the key if it doesn't exist.
                if key not in self.blocks:
                    self.blocks[key] = {
                        "t": [],
                        "aoi": [],
                        "v_block": [],
                        "ego_action": [],
                        "msg_sender": [],
                        "msg_id":[],
                        "ego_info": {
                            "cr_cn": value.get("ego_info", {}).get("cr_cn"),
                            "cr_cf": value.get("ego_info", {}).get("cr_cf"),
                            "gt_front": [],
                            "gt_rear": [],
                            "gt_pos": [],
                            "pred_go_pos": [],
                            "dist_to_cr": [],
                            "ttcr": [],
                            "ttlcr": [],
                            "ttlcr_less_adv_ttcr": [],
                            "ttcr_greater_adv_ttlcr": [],
                            "ttcr_less_adv_ttlcr": []
                        },
                        "target_info": {
                            "cr_cn": value.get("target_info", {}).get("cr_cn"),
                            "cr_cf": value.get("target_info", {}).get("cr_cf"),
                            "dist_to_cr": [],
                            "ttcr": [],
                            "ttlcr": [],
                            "pred_front": [],
                            "pred_rear": [],
                            "pred_rel_pos": [],
                            "gt_rel_pos": [],
                            "gt_front": [],
                            "gt_rear": [],
                        }
                    }
                # Append top-level values.
                self.blocks[key]["t"].append(value.get("sim_time"))
                self.blocks[key]["aoi"].append(value.get("aoi"))
                self.blocks[key]["v_block"].append(value.get("v_block"))
                self.blocks[key]["ego_action"].append(value.get("ego_action"))
                self.blocks[key]["msg_sender"].append(value.get("msg_sender"))
                self.blocks[key]["msg_id"].append(value.get("msg_id"))

                # Append values from the nested 'ego_info' dictionary.
                ego_info = value.get("ego_info", {})

                self.blocks[key]["ego_info"]["gt_front"].append(ego_info.get("gt_front"))
                self.blocks[key]["ego_info"]["gt_rear"].append(ego_info.get("gt_rear"))
                self.blocks[key]["ego_info"]["gt_pos"].append(ego_info.get("gt_current_pos"))
                self.blocks[key]["ego_info"]["pred_go_pos"].append(ego_info.get("pred_go_pos"))
                self.blocks[key]["ego_info"]["dist_to_cr"].append(ego_info.get("dist_to_cr"))
                self.blocks[key]["ego_info"]["ttcr"].append(ego_info.get("ttcr"))
                self.blocks[key]["ego_info"]["ttlcr"].append(ego_info.get("ttlcr"))
                self.blocks[key]["ego_info"]["ttlcr_less_adv_ttcr"].append(ego_info.get("ttlcr_less_adv_ttcr"))
                self.blocks[key]["ego_info"]["ttcr_greater_adv_ttlcr"].append(ego_info.get("ttcr_greater_adv_ttlcr"))
                self.blocks[key]["ego_info"]["ttcr_less_adv_ttlcr"].append(ego_info.get("ttcr_less_adv_ttlcr"))

                # Append values from the nested 'target_info' dictionary.
                target_info = value.get("target_info", {})
                self.blocks[key]["target_info"]["dist_to_cr"].append(target_info.get("dist_to_cr"))
                self.blocks[key]["target_info"]["ttcr"].append(target_info.get("ttcr"))
                self.blocks[key]["target_info"]["ttlcr"].append(target_info.get("ttlcr"))
                self.blocks[key]["target_info"]["pred_front"].append(target_info.get("pred_front"))
                self.blocks[key]["target_info"]["pred_rear"].append(target_info.get("pred_rear"))
                self.blocks[key]["target_info"]["pred_rel_pos"].append(target_info.get("pred_rel_pos"))
                self.blocks[key]["target_info"]["gt_rel_pos"].append(target_info.get("gt_rel_pos"))
                self.blocks[key]["target_info"]["gt_front"].append(target_info.get("gt_front"))
                self.blocks[key]["target_info"]["gt_rear"].append(target_info.get("gt_rear"))


class EgoData:
    def __init__(self):
        self.ac = list()
        self.ac_x = list()
        self.ac_y = list()
        self.ac_z = list()
        self.vel = list()
        self.v_x = list()
        self.v_y = list()
        self.pos_x = list()
        self.pos_y = list()
        self.fw_x = list()
        self.fw_y = list()
        self.rg_x = list()
        self.rg_y = list()
        self.dist_to_obs = list()
        self.v_sel = list()
        self.v_obs = list()

    def add(self, j_data):
        self.ac.append(j_data["ac"])
        self.ac_x.append(j_data["ac_x"])
        self.ac_y.append(j_data["ac_y"])
        self.ac_z.append(j_data["ac_z"])
        self.vel.append(j_data["vel"])
        self.v_x.append(j_data["v_x"])
        self.v_y.append(j_data["v_y"])
        self.pos_x.append(j_data["pos_x"])
        self.pos_y.append(j_data["pos_y"])
        self.fw_x.append(j_data["fw_x"])
        self.fw_y.append(j_data["fw_y"])
        self.rg_x.append(j_data["rg_x"])
        self.rg_y.append(j_data["rg_y"])
        self.dist_to_obs.append(j_data["dist_to_obs"])
        self.v_sel.append(j_data["v_selected"])
        self.v_obs.append(j_data["v_obs"])


class CollisionData:
    def __init__(self):
        self.t = list()
        self.value = list()
        self.impulse_x = list()
        self.impulse_y = list()
        self.impulse_z = list()

    def add(self, j_data):
        self.t.append(j_data["t"])
        self.value.append(j_data["value"])
        self.impulse_x.append(j_data["impulse_x"])
        self.impulse_y.append(j_data["impulse_y"])
        self.impulse_z.append(j_data["impulse_z"])


class ImuData:
    def __init__(self):
        self.a = list()
        self.a_x = list()
        self.a_y = list()
        self.a_z = list()

    def add(self, j_data):
        self.a.append(j_data["a"] if j_data["a"] < 1000 else 0)
        self.a_x.append(j_data["a_x"] if j_data["a_x"] < 1000 else 0)
        self.a_y.append(j_data["a_y"] if j_data["a_y"] < 1000 else 0)
        self.a_z.append(j_data["a_z"] if j_data["a_z"] < 1000 else 0)


class EgoIncomingMsg:
    def __init__(self):
        self.msgs = list()

    def add(self, j_data):
        data = {"t": j_data["t"], "sender": j_data["sender"], "id": j_data["id"]}
        self.msgs.append(data)


class SimScenarioData:
    def __init__(self):
        self.t_start = -1
        self.t_end = -1
        self.delta_t = -1
        self.cause = None
        self.name = None

    def add(self, j_data):
        self.t_start = j_data["t_start"]
        self.t_end =   j_data["t_end"]
        self.delta_t = j_data["delta_t"]
        self.cause =   j_data["cause"]
        self.name =    j_data["scenario_name"]


class RedisScenarioTerminationStream:
    def __init__(self):
        self.data = SimScenarioData()
        self.last_id = "-"

    def get_data(self, h_set_exp, sim_term_key):
        # get the session key
        stream_name = rd.get_stream_name(h_set_exp, sim_term_key)
        res = rd.redis_instance.xrange(stream_name, self.last_id, "+")

        for item in res:
            self.last_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.data.add(j_data)


class RedisComStream:

    def __init__(self):
        self.data = CommData()
        self.last_id = "-"

    def get_data(self, h_set_exp, com_key):
        # get the session key
        stream_name = rd.get_stream_name(h_set_exp, com_key)
        res = rd.redis_instance.xrange(stream_name, self.last_id, "+")

        for item in res:
            self.last_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.data.add(j_data)


class RedisExtSensorStream:

    def __init__(self):
        self.data = ExtSensorData()
        self.last_ids = {}

    def get_data(self, h_set_exp, ext_sensor_key):
        # get the session key
        stream_name = rd.get_stream_name(h_set_exp, ext_sensor_key)

        for e in rd.redis_instance.keys(stream_name + ":*"):
            if e not in self.last_ids:
                self.last_ids[e] = "-"

            res = rd.redis_instance.xrange(e, self.last_ids[e], "+")
            for item in res:
                self.last_ids[e] = item[0].decode()
                j_data = json.loads(item[1][b'data'])
                self.data.add(j_data)


class RedisEgoStreams:

    def __init__(self):
        self.data = EgoData()
        self.tactical_data = TacticalData()
        self.perception_data = PerceptionData()
        self.ext_msg_data = EgoIncomingMsg()
        self.imu_data = ImuData()
        self.coll_data = CollisionData()
        self.last_perc_id = "-"
        self.last_risk_id = "-"
        self.last_data_id = "-"
        self.last_msg_id = "-"
        self.last_imu_id = "-"
        self.last_coll_id = "-"


    def get_perception_data(self, h_set_exp, risk_key):
        # get the session key
        stream_name = rd.get_stream_name(h_set_exp, risk_key)
        res = rd.redis_instance.xrange(stream_name, self.last_perc_id, "+")

        for item in res:
            self.last_perc_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.perception_data.add(j_data)

    def get_tactical_data(self, h_set_exp, risk_key):
        # get the session key
        stream_name = rd.get_stream_name(h_set_exp, risk_key)
        res = rd.redis_instance.xrange(stream_name, self.last_risk_id, "+")

        for item in res:
            self.last_risk_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.tactical_data.add(j_data)

    def get_ext_msg_data(self, h_set_exp, key):

        stream_name = rd.get_stream_name(h_set_exp, key)
        res = rd.redis_instance.xrange(stream_name, self.last_msg_id, "+")

        for item in res:
            self.last_msg_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.ext_msg_data.add(j_data)

    def get_collision_data(self, h_set_exp, key):

        stream_name = rd.get_stream_name(h_set_exp, key)
        res = rd.redis_instance.xrange(stream_name, self.last_coll_id, "+")

        for item in res:
            self.last_coll_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.coll_data.add(j_data)

    def get_imu_data(self, h_set_exp, key):

        stream_name = rd.get_stream_name(h_set_exp, key)
        res = rd.redis_instance.xrange(stream_name, self.last_imu_id, "+")

        for item in res:
            self.last_imu_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.imu_data.add(j_data)

    def get_ego_data(self, h_set_exp, key):

        stream_name = rd.get_stream_name(h_set_exp, key)
        res = rd.redis_instance.xrange(stream_name, self.last_data_id, "+")

        for item in res:
            self.last_data_id = item[0].decode()
            j_data = json.loads(item[1][b'data'])
            self.data.add(j_data)



class RedisStreamEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class RedisSimulationStream:
    def __init__(self):
        self.ego_stream = RedisEgoStreams()
        self.adv_stream = RedisEgoStreams()
        self.com_stream = RedisComStream()
        self.ext_sensor_stream = RedisExtSensorStream()
        self.scenario_term_stream = RedisScenarioTerminationStream()

    def get_data(self, exp_name: str):
        key = rd.get_h_key(exp_name, rd.K_EGO_DATA)
        self.ego_stream.get_ego_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EGO_IMU)
        self.ego_stream.get_imu_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EGO_COL)
        self.ego_stream.get_collision_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EGO_TACTICAL)
        self.ego_stream.get_tactical_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EGO_PERCEPT)
        self.ego_stream.get_perception_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EGO_MSG)
        self.ego_stream.get_ext_msg_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_DATA)
        self.adv_stream.get_ego_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_IMU)
        self.adv_stream.get_imu_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_COL)
        self.adv_stream.get_collision_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_TACTICAL)
        self.adv_stream.get_tactical_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_PERCEPT)
        self.adv_stream.get_perception_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_ADVERSARY_MSG)
        self.adv_stream.get_ext_msg_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_COM)
        self.com_stream.get_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EXT_SENSOR_MSG)
        self.ext_sensor_stream.get_data(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_SCENARIO_TERMINATION)
        self.scenario_term_stream.get_data(exp_name, key.decode())

    def write_to_file(self, filename, rare_event_data=None, execution_data=None):
        ego_data = {"data": self.ego_stream.data,
                    "perception_data": self.ego_stream.perception_data,
                    "tactical_data": self.ego_stream.tactical_data,
                    "coll_data": self.ego_stream.coll_data,
                    "ext_msg_data": self.ego_stream.ext_msg_data}

        adv_data = {"data": self.adv_stream.data,
                    "perception_data": self.adv_stream.perception_data,
                    "tactical_data": self.adv_stream.tactical_data,
                    "coll_data": self.adv_stream.coll_data,
                    "ext_msg_data": self.adv_stream.ext_msg_data}

        data = {"version": 6,
                "scenario": self.scenario_term_stream.data,
                "ego": ego_data,
                "adv": adv_data,
                "obps": self.ext_sensor_stream.data,
                "comm": self.com_stream.data
                }

        if execution_data is not None:
            data["execution"] = execution_data
        
        if rare_event_data is not None:
            data["rare_event"] = rare_event_data

        with open(filename, 'w') as f:
            json.dump(data, f, cls=SimulationStreamEncoder)

        s = json.dumps(data, cls=SimulationStreamEncoder)

        dict_out = json.loads(s)

        return dict_out

    @staticmethod
    def delete_streams(exp_name: str):
        def delete(h_set_exp, k):
            stream_name = rd.get_stream_name(h_set_exp, k)
            res = rd.redis_instance.xrange(stream_name, "-", "+")
            for item in res:
                message_id = item[0].decode()
                rd.redis_instance.xdel(stream_name, message_id)

        for k in rd.EGO_KEYS:
            key = rd.get_h_key(exp_name, k)
            delete(exp_name, key.decode())

        for k in rd.ADV_KEYS:
            key = rd.get_h_key(exp_name, k)
            delete(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_COM)
        delete(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_EXT_SENSOR_MSG)
        delete(exp_name, key.decode())

        key = rd.get_h_key(exp_name, rd.K_SCENARIO_TERMINATION)
        delete(exp_name, key.decode())


class SimulationStreamEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
