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
from typing import List, Optional


from source.simulation_environment.obps.message import SensorMsg
from source.simulation_environment.logs.com_logs import ComLogItem
import source.simulation_environment.logs.printl
import source.simulation_environment.logs.redis_data as rd
from source.simulation_environment.communication.network_models import network_models_factory
from source.simulation_environment.communication.fault_models import fault_models_factory
from source.simulation_environment.communication.fault_models.fault_logic_base import ComEvent
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    require_sampler_for_bias,
)


class Communication:
    TAG = "Comm"

    class Item:
        def __init__(self, t, payload: SensorMsg, age):
            self.payload = payload
            self.age = age
            self.ingress_time = t

        def get_aging(self):
            return self.ingress_time + self.age

    def __init__(self, delta_time: float, exp_name: str, cfg: dict,
                 sampler: Optional[ImportanceSampler] = None,
                 bias_cfg: Optional[dict] = None):
        """

        :param delta_time:
        :param exp_name:
        :param cfg: must be the "communication" block
        """
        require_sampler_for_bias(sampler, bias_cfg, "Communication")
        self.memory = {}
        self._tick = 0
        self._clock = 0
        self.online = True
        self.fault_model = None
        self.network_model = network_models_factory.get_network_model(cfg["network"],
                                                                      sampler=sampler,
                                                                      bias_cfg=bias_cfg.get("network_bias") if bias_cfg else None)
        self.fault_model = fault_models_factory.get_fault_model(delta_time, cfg["fault"], sampler=sampler,
                                                                bias_cfg=bias_cfg.get("fault_bias") if bias_cfg else None)
        self.dropped_packets = list()

        self.rd_stream_name = rd.get_stream_name(exp_name, rd.get_h_key(exp_name, rd.K_COM).decode())
        self.printl = source.simulation_environment.logs.printl.PrintL(Communication.TAG, enabled=True)


    def ingress(self, sim_time, msg: SensorMsg, recipient=0) -> bool:

        if not self.online:
            return False

        age = self.network_model.compute_De2e(sim_time=sim_time)

        # check if the communication will drop the packet, return true as the packet was "processed"
        # by the communication network
        if self.network_model.should_drop_packet(sim_time=sim_time):
            self.dropped_packets.append({"sim_time":sim_time ,"sender":msg.sender, "id":msg.id})
            return True

        if recipient in self.memory:
            self.memory[recipient].append(Communication.Item(self._clock, msg, age))
        else:
            self.memory[recipient] = list()
            self.memory[recipient].append(Communication.Item(self._clock, msg, age))

        return True

    def get_messages(self, receiver=0) -> List[SensorMsg]:
        if not self.online:
            return []

        ready_msg = []
        try:
            ready_msg = [x for x in self.memory[receiver] if self._clock >= x.get_aging()]
            self.memory[receiver] = list(set(self.memory[receiver]) - set(ready_msg))
            ready_msg = [x.payload for x in ready_msg]
        except KeyError:
            self.printl.to_print(header="EVENT", message="receiver key not present, no message available")

        return ready_msg

    def parse_event(self, event: ComEvent):
        if event == event.OFFLINE:
            if self.online:
                self.printl.to_print(header="EVENT", message="going off-line")
            self.online = False
        elif event == event.ONLINE:
            if not self.online:
                self.printl.to_print(header="EVENT", message="going on-line")
            self.online = True

    def comm_tick(self, sim_time):
        self._clock = sim_time
        event = self.fault_model.step(sim_time)
        self.parse_event(event)
        self.log(sim_time)

    def log(self, sim_time):
        item = ComLogItem(sim_time, self.online, self.dropped_packets)
        rd.save_stream_com(self.rd_stream_name, item)
        self.dropped_packets.clear()
