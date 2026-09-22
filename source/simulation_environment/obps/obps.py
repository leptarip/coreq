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
from __future__ import annotations

from collections import deque
from typing import Optional

import carla

from source.simulation_environment.communication.communication import Communication
from source.simulation_environment.obps.logic.static_ghost import SectorStaticGhost

from source.simulation_environment.obps.message import SensorMsg, SensorMsgContent
from source.simulation_environment.cfg.sensor_cfg import SensorCfg
from source.simulation_environment.obps.logic.sector_logic import SectorLogic
from source.simulation_environment.rare_events.wrapper import ImportanceSampler
import source.simulation_environment.utils as utils
from source.simulation_environment.obps.sensor_result import SensorResult
from source.simulation_environment.logs.ext_sensor_log import ExtSensorLogItem
import source.simulation_environment.logs.redis_data as rd
import source.simulation_environment.logs.printl
import source.simulation_environment.sim_utils as sim_utils


class Sensor:
    TAG = "Sensor"

    GROUND_TRUTH_NO_TARGET = 0
    GROUND_TRUTH = 1
    GHOST = 2
    MISS = 3

    def __init__(self,
                 log_key,
                 cfg: SensorCfg,
                 comm_medium: Communication,
                 target: carla.Actor,
                 delta_time: float,
                 sensor_bias: Optional[dict] = None,
                 sampler: Optional["ImportanceSampler"] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.t_gen = cfg.t_gen
        self._sim_time = 0
        self._gen_clock = 0
        self.unit_step = sim_utils.get_sim_step(delta_time) #get the step in ms
        self.target = target
        self.comm_medium = comm_medium
        self.name = utils.create_sensor_name(cfg.name)
        self.stream_msg_name = utils.create_sensor_msg_key(log_key, self.name)
        self.msg_count = 0
        self.recipient = cfg.recipient
        self.stream = cfg.stream
        self.sensor_result: Optional[SensorResult] = None

        # optional bias + sampler for rare-event runs
        self.sensor_bias = sensor_bias or {}
        self.sampler = sampler
        self.seed = seed

        self.logic_type = cfg.fov.type_id
        if self.logic_type == cfg.TYPE_SECTOR:
            self.sensor_logic = SectorLogic(cfg.fov, sensor_bias=self.sensor_bias, sampler=self.sampler, seed=self.seed)
        elif self.logic_type == cfg.TYPE_SECTOR_GHOST:
            self.sensor_logic = SectorStaticGhost(cfg.fov, seed=self.seed)

        self.printl = source.simulation_environment.logs.printl.PrintL(Sensor.TAG, enabled=True)
        self.message_sent = False

        if self.cfg.retransmission_queue_size > 0:
            self.retransmission_queue = deque(maxlen=self.cfg.retransmission_queue_size)


    def get_position(self):
        return self.cfg.fov.position

    def get_orientation(self):
        return self.cfg.fov.orientation

    def gen_observation(self):
        if self._gen_clock >= self.t_gen:
            self._gen_clock = 0
            return True
        return False

    def detect(self, target, actor_shapes, sim_time=None, shape_by_actor_id=None) -> SensorResult:
        result = self.sensor_logic.logic_step(target, actor_shapes, sim_time=sim_time,
                                              shape_by_actor_id=shape_by_actor_id)
        return result

    def generate_msg(self, tick, result: SensorResult):

        metadata = SensorMsgContent.MetaData(result.fov_region, result.sensor_prob)

        msg = SensorMsg(self.name,
                        self.msg_count,
                        tick,
                        SensorMsgContent(meta_data=metadata, measurement=result.measurement)
                        )
        self.msg_count = self.msg_count + 1
        return msg

    """
    The retransmission policy should be parameterizable in the configuration
    1) the old message is always retransmitted first
    2) the new message is always retransmitted first
    We should guarantee that 1) or 2) is fulfilled all the times
    In this implementation there is no guarantee that the retransmission happens following 
    an order.
    NOTE: The current implementation provides a circular queue for retransmission
    which does not guarantee 1) or 2)
    """
    def send_msg(self, sim_time, msg) -> bool:
        ret = self.comm_medium.ingress(sim_time, msg, self.recipient)
        if ret:
            self.printl.to_print(header="{0}".format(msg.sender),
                                 message="Sent message_id: {0}, sim_time: {1}".format(msg.id, self._sim_time)
                                 )
            self.message_sent = True
        else:
            if len(self.retransmission_queue) >= self.cfg.retransmission_queue_size:
                self.retransmission_queue.popleft()
            self.retransmission_queue.append(msg)
            self.message_sent = False
            self.printl.to_print(header="{0}".format(msg.sender),
                                 message="Communication offline"
                                 )
        return ret

    def sensor_tick(self, sim_time, actor_shapes, shape_by_actor_id=None):
        self._sim_time = sim_time
        self._gen_clock = self._gen_clock + self.unit_step

        # retransmission of messages
        for _ in range(len(self.retransmission_queue)):
            if self.cfg.retransmission_queue_policy == "lifo":
                ret = self.send_msg(sim_time, self.retransmission_queue.pop())
            else:
                ret = self.send_msg(sim_time, self.retransmission_queue.popleft())
            if not ret:
                break

        # check if it is time to generate an observation
        if self.gen_observation():
            # check if we detect something
            self.sensor_result = self.detect(self.target, actor_shapes, sim_time=sim_time,
                                             shape_by_actor_id=shape_by_actor_id)
            self._log_message(sim_time)

            if self.sensor_result.result != SensorResult.NO_DETECTION and self.sensor_result.result != SensorResult.MISS:
                msg = self.generate_msg(self._sim_time, self.sensor_result)
                self.send_msg(sim_time, msg)

    def _log_message(self, sim_time):
        msg_id = -1
        if self.sensor_result.result != SensorResult.NO_DETECTION and self.sensor_result.result != SensorResult.MISS:
            msg_id = self.msg_count

        msg_log_data = ExtSensorLogItem(
            sim_time,
            self.name,
            msg_id,
            self.sensor_result.result_to_string(),
            self.sensor_result.measurement,
            self.sensor_result.fov_region_name
        )
        rd.save_stream_ext_sensor(self.stream_msg_name, msg_log_data)

    ##if we sent a message including retransmission (multiple messages could have been sent!)
    def is_msg_sent(self):
        return self.message_sent
